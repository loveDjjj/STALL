# Duration / window 敏感性可行性审计

本审计只读取 index CSV 与已有 compact patch cache，不重新提取 DINOv3 特征。
结论用于决定是否值得立即实跑 1s/2s/3s/4s duration 敏感性实验。

## 数据集级结论

| dataset | duration | real indexed | fake indexed | real cache | fake cache | 可直接 patch eval | 说明 |
|---|---:|---:|---:|---:|---:|---|---|
| comgenvid | 1s | 1700 | 3400 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| comgenvid | 2s | 1698 | 3400 | 1698 | 3400 | yes | 已有真实与生成 compact patch cache，可直接创建该 duration 的 patch params 并评测。 |
| comgenvid | 3s | 1693 | 3400 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| comgenvid | 4s | 1639 | 3400 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| genvideo | 1s | 9984 | 8204 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| genvideo | 2s | 9984 | 5639 | 9984 | 5639 | yes | 已有真实与生成 compact patch cache，可直接创建该 duration 的 patch params 并评测。 |
| genvideo | 3s | 9984 | 3109 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| genvideo | 4s | 9984 | 2341 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| videofeedback | 1s | 4080 | 33581 | 0 | 3251 | no | 只有生成 cache，缺真实 cache，不能做真实视频校准。 |
| videofeedback | 2s | 4080 | 30330 | 4080 | 30330 | yes | 已有真实与生成 compact patch cache，可直接创建该 duration 的 patch params 并评测。 |
| videofeedback | 3s | 2703 | 8266 | 0 | 0 | no | index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。 |
| videofeedback | 4s | 0 | 0 | 0 | 0 | no | index 无该时长窗口；需重新定义窗口或更换数据。 |

## 关键判断

- 当前三数据集只有 2s compact patch cache 达到主实验可用覆盖，因此主线 2s 结果是唯一已经完整落地的 duration。
- VideoFeedback 虽有 1s index 全覆盖，但已有 1s patch cache 只覆盖 Hotshot-XL 生成视频，缺真实视频 cache，不能进行真实视频校准。
- GenVideo 与 ComGenVid 的 index 支持 1s/3s/4s 窗口，但 compact patch cache 目前只存在 2s，因此直接实跑 duration sweep 会触发大规模 cache prefill。
- 若需要投期刊前补 duration/window 敏感性，建议先只补一个代表数据集的 1s/2s 对照，并把 3s/4s 作为计算预算较高的扩展实验。

## VideoFeedback 1s cache 覆盖明细

| subset/source | indexed 1s | cached 1s | coverage |
|---|---:|---:|---:|
| annotated/AnimateDiff | 1399 | 0 | 0.000 |
| annotated/Fast-SVD | 1000 | 0 | 0.000 |
| annotated/Hotshot-XL | 3251 | 3251 | 1.000 |
| annotated/LVDM | 3172 | 0 | 0.000 |
| annotated/LaVie-base | 3214 | 0 | 0.000 |
| annotated/ModelScope | 4565 | 0 | 0.000 |
| annotated/Pika | 4644 | 0 | 0.000 |
| annotated/SoRA-Clip | 920 | 0 | 0.000 |
| annotated/Text2Video-Zero | 4642 | 0 | 0.000 |
| annotated/VideoCrafter2 | 4596 | 0 | 0.000 |
| annotated/ZeroScope-576w | 2178 | 0 | 0.000 |
| real/DiDeMo | 1861 | 0 | 0.000 |
| real/Panda70M | 2219 | 0 | 0.000 |
