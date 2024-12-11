# SDXL with LitServe

See [LitServe で画像生成サーバーを建てるGitHubで開く](https://zenn.dev/platina/articles/6765e3d74b0f1b)

## Setup

```bash
git clone https://github.com/p1atdev/sdxl-litserve.git
cd sdxl-litserve
uv sync
```

## Serve

```bash
python ./txt2image.py
```

## Client

```bash
curl --location 'http://localhost:8000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "1girl, red eyes, blush, twintails, smile, open mouth, open mouth, sitting, hair ornament, orange hair, pov, dress, sparkle, bow, animal ears, hair between eyes, black dress, black bow, sidelocks, blurry, :d, chibi, hair bow, from above, detached sleeves, fang, headpat, blurry foreground, hair bobbles, wings, indoors, detached collar, skin fang, skin fang, animal ear fluff, pov hands, dress bow, wooden floor, animal, rabbit, grey fur, bat ears, bat girl",
    "width": 1024,
    "height": 1024,
    "cfg_scale": 6.5,
    "inference_steps": 25
}' \
--output ./output.webp
```


