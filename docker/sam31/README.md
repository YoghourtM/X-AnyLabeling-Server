# Local SAM 3.1 server

Serves `segment_anything_3_1` to X-AnyLabeling over the Remote-Server
protocol. All three tasks share one `sam3.1_multiplex.pt` checkpoint:

| Task | Input | Output |
| --- | --- | --- |
| Text prompt | Concepts such as `person, car` | One polygon per instance |
| Click / box (single object) | Positive / negative clicks, or one box | One polygon |
| Example boxes (find similar) | Positive / negative example boxes | Polygons for similar objects |

By default (`detector: shared`) one SAM 3.1 multiplex model serves every
task in about 5.5 GB of GPU memory: its detector handles text prompts and
example boxes, its tracker handles clicks. `detector: image_model` instead
runs text and example boxes on a separate SAM 3 image model carrying the 3.1
detector weights (the batch pipelines' setup, scores about 0.1 higher), which
needs about 4 GB more once clicks are used. Video tracking is not provided.

## Run

```bash
export SAM31_CHECKPOINT=/path/to/sam3.1_multiplex.pt
scripts/sam31_server.sh build    # once; needs the SAM 3.1 base image
scripts/sam31_server.sh start    # http://127.0.0.1:8000
scripts/sam31_server.sh status
scripts/sam31_server.sh logs
scripts/sam31_server.sh stop
```

`BASE_IMAGE`, `SAM31_PORT` and `SAM31_IMAGE` override the defaults. The
container listens on 127.0.0.1 only, runs as the calling user and mounts the
working tree at /app, so code changes need only a restart.

In X-AnyLabeling, choose **Remote-Server**, set the server URL to
`http://127.0.0.1:8000`, then pick **Segment Anything 3.1** and a task.
