import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

_BACKEND = Path(__file__).resolve().parents[1]
_SCRATCH = tempfile.mkdtemp(prefix='gb0071-test-')
os.environ['DATABASE_URL'] = 'sqlite:///' + os.path.join(_SCRATCH, 'test.db')
os.environ['REDIS_URL'] = 'redis://127.0.0.1:6390/0'
os.environ['ARTIFACTS_DIR'] = _SCRATCH
sys.path.insert(0, str(_BACKEND))

from sqlalchemy import insert, select  # noqa: E402

from src import worker as worker_module  # noqa: E402
from src.models.database import engine  # noqa: E402
from src.models.tables import audit, datasets, metadata, pair_distances, runs  # noqa: E402
from src.services import distances as distance_service  # noqa: E402
from src.services.data import save_revision  # noqa: E402
from src.services.distances import resolve_all  # noqa: E402
from src.services.domain import _dtw, assemble, compute, plan, sample_rows  # noqa: E402
from src.services.workflow import add_run, claim_run  # noqa: E402

OPTIONS = {'cluster_count': 3, 'warp_band': 1}


def spy_on(calls):
    def spy(left, right, band):
        calls.append((left, right, band))
        return _dtw(left, right, band)
    return spy


def pair_row(key):
    with engine.connect() as connection:
        return connection.execute(select(pair_distances).where(pair_distances.c.pair_key == key)).mappings().one()


class IncrementalDistanceTests(unittest.TestCase):
    def setUp(self):
        metadata.create_all(engine)
        with engine.begin() as connection:
            for table in (pair_distances, runs, datasets, audit):
                connection.execute(table.delete())

    def test_corrected_batch_reuses_matching_pairs_and_matches_scratch(self):
        rows = sample_rows()
        # 夜充批次更正：C1124-09 补上缺失的 15 分钟点，其余会话不变。
        broken = [row for row in rows if not (row['session_id'] == 'C1124-09' and row['minute_offset'] == 15)]
        planned_v1 = plan(broken, OPTIONS)
        self.assertEqual(len(planned_v1['complete_ids']), 11)

        calls_v1 = []
        found_v1 = resolve_all(planned_v1['pairs'], planned_v1['band'], 'run-1', compute=spy_on(calls_v1))
        self.assertEqual(len(calls_v1), len(planned_v1['pairs']))

        planned_v2 = plan(rows, OPTIONS)
        self.assertEqual(len(planned_v2['complete_ids']), 12)
        new_keys = set(planned_v2['pairs']) - set(planned_v1['pairs'])
        self.assertEqual(len(new_keys), 11)

        calls_v2 = []
        found_v2 = resolve_all(planned_v2['pairs'], planned_v2['band'], 'run-2', compute=spy_on(calls_v2))
        # 完整曲线和带宽都相同的距离沿用，只有涉及更正会话的 11 对重算。
        self.assertEqual(len(calls_v2), len(new_keys))
        corrected_curve = planned_v2['sessions']['C1124-09'].curve()
        for left, right, band in calls_v2:
            self.assertEqual(band, OPTIONS['warp_band'])
            self.assertTrue(corrected_curve in (left, right))
        for key in planned_v1['pairs']:
            self.assertEqual(found_v2[key], found_v1[key])

        # 中心和归属要与从头算一致。
        self.assertEqual(assemble(planned_v2, found_v2), compute(rows, OPTIONS))

    def test_concurrent_batches_dispatch_each_pair_once(self):
        planned = plan(sample_rows(), OPTIONS)
        total = len(planned['pairs'])
        calls = []
        outcomes = []
        errors = []
        barrier = threading.Barrier(2)
        lock = threading.Lock()

        def slow(left, right, band):
            time.sleep(0.02)
            value = _dtw(left, right, band)
            with lock:
                calls.append((left, right, band))
            return value

        def batch(tag):
            try:
                barrier.wait(timeout=5)
                outcomes.append(resolve_all(planned['pairs'], planned['band'], tag,
                                            compute=slow, sleep=lambda _: time.sleep(0.01)))
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=batch, args=(f'run-{number}',)) for number in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual(errors, [])
        # 同一对曲线和同一带宽只派一份计算。
        self.assertEqual(len(calls), total)
        self.assertEqual(len({key for outcome in outcomes for key in outcome}), total)
        self.assertEqual(outcomes[0], outcomes[1])

    def test_interrupted_batch_resumes_and_takes_over_stale_claims(self):
        planned = plan(sample_rows(), OPTIONS)
        items = list(planned['pairs'].items())
        # 中断前已保存 20 对距离。
        resolve_all(dict(items[:20]), planned['band'], 'run-old')
        # 中断时还有一对停留在认领中，认领已过期。
        stale_key, stale_curves = items[20]
        stale_at = time.time() - 3600
        with engine.begin() as connection:
            connection.execute(insert(pair_distances).values(pair_key=stale_key, band=planned['band'],
                                                             state='pending', claim='old-token',
                                                             claimed_by='run-old', claimed_at=stale_at,
                                                             created=stale_at))

        calls = []
        found = resolve_all(planned['pairs'], planned['band'], 'run-new', compute=spy_on(calls))

        # 已经保存的距离不用重算，等这对距离的批次接手过期认领后续跑。
        self.assertEqual(len(calls), len(items) - 20)
        self.assertIn((stale_curves[0], stale_curves[1], planned['band']), calls)
        row = pair_row(stale_key)
        self.assertEqual(row['state'], 'done')
        self.assertEqual(row['claimed_by'], 'run-new')
        self.assertEqual(row['distance'], found[stale_key])
        self.assertEqual(len(found), len(items))

    def test_late_claim_cannot_overwrite_taken_over_result(self):
        planned = plan(sample_rows(), OPTIONS)
        key, curves = next(iter(planned['pairs'].items()))
        stale_at = time.time() - 3600
        with engine.begin() as connection:
            connection.execute(insert(pair_distances).values(pair_key=key, band=planned['band'],
                                                             state='pending', claim='old-token',
                                                             claimed_by='run-old', claimed_at=stale_at,
                                                             created=stale_at))
        found = resolve_all({key: curves}, planned['band'], 'run-new')
        value = found[key]

        # 旧计算迟到，认领已被接手，不能覆盖结果。
        self.assertFalse(distance_service._publish(key, 'old-token', value + 999, time.time()))
        row = pair_row(key)
        self.assertEqual(row['state'], 'done')
        self.assertEqual(row['distance'], value)

        # 认领中的对也不接受无关令牌的结果。
        other_key, _ = list(planned['pairs'].items())[1]
        with engine.begin() as connection:
            connection.execute(insert(pair_distances).values(pair_key=other_key, band=planned['band'],
                                                             state='pending', claim='live-token',
                                                             claimed_by='run-live', claimed_at=time.time(),
                                                             created=time.time()))
        self.assertFalse(distance_service._publish(other_key, 'wrong-token', 1.0, time.time()))
        pending = pair_row(other_key)
        self.assertEqual(pending['state'], 'pending')
        self.assertIsNone(pending['distance'])

    def test_assignments_visible_only_after_all_distances(self):
        records = sample_rows()
        dataset_id, _ = save_revision('夜班更正批', records, 'analyst')
        queued = add_run(SimpleNamespace(dataset_id=dataset_id, options=dict(OPTIONS)), 'analyst')
        job = claim_run()
        self.assertEqual(job[0], queued['id'])

        seen = {}
        real_resolve = worker_module.resolve_all

        def checking_resolve(pairs, band, owner):
            with engine.connect() as connection:
                row = connection.execute(select(runs.c.state, runs.c.result)
                                         .where(runs.c.id == queued['id'])).mappings().one()
            seen['state'] = row['state']
            seen['result'] = row['result']
            return real_resolve(pairs, band, owner)

        worker_module.resolve_all = checking_resolve
        try:
            worker_module.calculate(job)
        finally:
            worker_module.resolve_all = real_resolve

        # 距离未齐全时，页面读到的任务没有结果，不显示归属。
        self.assertEqual(seen['state'], 'running')
        self.assertIsNone(seen['result'])

        with engine.connect() as connection:
            done = connection.execute(select(runs).where(runs.c.id == queued['id'])).mappings().one()
        self.assertEqual(done['state'], 'completed')
        result = done['result']
        self.assertIsNotNone(result)
        # 距离齐全后一次性给出整批归属，且与从头算一致。
        expected = compute(records, OPTIONS)
        for field in ('headline', 'metrics', 'rows', 'notes', 'chart', 'can_publish'):
            self.assertEqual(result[field], expected[field])
        self.assertEqual(len([row for row in result['rows'] if row['所属分群']]), 12)

    def test_failed_batch_shows_no_assignments(self):
        records = [row for row in sample_rows() if row['session_id'] in ('C1124-01', 'C1124-02')]
        dataset_id, _ = save_revision('完整会话不足批', records, 'analyst')
        queued = add_run(SimpleNamespace(dataset_id=dataset_id, options=dict(OPTIONS)), 'analyst')
        worker_module.calculate(claim_run())
        with engine.connect() as connection:
            done = connection.execute(select(runs).where(runs.c.id == queued['id'])).mappings().one()
        self.assertEqual(done['state'], 'failed')
        self.assertIsNone(done['result'])
        self.assertIn('完整会话仅 2 条', done['error'])


if __name__ == '__main__':
    unittest.main()
