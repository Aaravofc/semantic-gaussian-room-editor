# Demo Guide

The checked-in demo is synthetic. It contains a small Gaussian scene and six semantic proxy
objects, so it can be published without exposing a private room capture.

## Record a portfolio clip

1. Start the viewer using the command in the main README.
2. Orbit around the scene and select the couch, table, lamp, and plant.
3. Filter the object list by label.
4. Move or rotate one proxy and save the edit state.
5. Record a 15–30 second clip at 1080p and place it at `docs/demo.mp4` or convert it to a GIF.

Keep the caption precise: proxy transforms are non-destructive metadata and do not yet move
the underlying Gaussian group.

## Replace the synthetic demo

Only publish a real scan when you own the capture and are comfortable disclosing its visual
appearance and spatial layout. Put larger assets in a GitHub Release or external object store
instead of normal Git history.

