import secrets
import time

from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from src.models.database import engine
from src.models.tables import pair_distances
from src.services.domain import _dtw

CLAIM_TTL = 60.0
POLL_SECONDS = 0.4


def _claim(key, band, owner, now):
    """Take ownership of one pair and return its fencing token, or None if another
    batch is already computing it. A pending pair whose claim went stale (its owner
    was interrupted) is taken over; a missing pair is dispatched exactly once."""
    token = secrets.token_hex(8)
    with engine.begin() as connection:
        taken = connection.execute(update(pair_distances)
                                   .where(pair_distances.c.pair_key == key,
                                          pair_distances.c.state == 'pending',
                                          pair_distances.c.claimed_at < now - CLAIM_TTL)
                                   .values(claim=token, claimed_by=owner, claimed_at=now))
        if taken.rowcount == 1:
            return token
    try:
        with engine.begin() as connection:
            connection.execute(insert(pair_distances).values(pair_key=key, band=band, state='pending',
                                                             claim=token, claimed_by=owner,
                                                             claimed_at=now, created=now))
        return token
    except IntegrityError:
        return None


def _publish(key, token, value, now):
    """Store the distance iff the pair is still ours. A late computation whose claim
    was taken over loses the fence and cannot overwrite the newer row."""
    with engine.begin() as connection:
        saved = connection.execute(update(pair_distances)
                                   .where(pair_distances.c.pair_key == key,
                                          pair_distances.c.claim == token,
                                          pair_distances.c.state == 'pending')
                                   .values(state='done', distance=value, finished=now))
        return saved.rowcount == 1


def resolve_all(pairs, band, owner, compute=_dtw, sleep=time.sleep):
    """Return {pair_key: distance} for every pair of a batch.

    Distances already stored for the same curve pair and band are reused as-is;
    each missing pair is computed by exactly one owner across concurrent batches,
    and pairs left pending by an interrupted computation are taken over.
    """
    resolved = {}
    pending = dict(pairs)
    while pending:
        with engine.connect() as connection:
            stored = {row['pair_key']: row for row in connection.execute(
                select(pair_distances.c.pair_key, pair_distances.c.state, pair_distances.c.distance)
                .where(pair_distances.c.pair_key.in_(list(pending)))).mappings()}
        for key in list(pending):
            row = stored.get(key)
            if row is not None and row['state'] == 'done':
                resolved[key] = row['distance']
                del pending[key]
                continue
            token = _claim(key, band, owner, time.time())
            if token is None:
                continue
            left, right = pending[key]
            value = compute(left, right, band)
            if _publish(key, token, value, time.time()):
                resolved[key] = value
                del pending[key]
        if pending:
            sleep(POLL_SECONDS)
    return resolved
