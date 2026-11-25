初始结构由 `src/models/tables.py` 定义，在 API 启动前由 `metadata.create_all` 创建。演示数据通过独立入口填充。

`curve_distances` 距离缓存表同样由 `create_all` 创建：对已有数据卷只是新增该表，无需改写既有表；删除该表只会丢失已缓存的曲线距离，重算后会自动回填。
