# O catálogo

**O que é isto.** Todo componente registrado e o que seus parâmetros
significam. Sources produzem frames, stages os transformam, sinks os
consomem; o formato que conecta tudo isso está em
[Escrevendo um config](configuration.md).

![O catálogo de stages, por família](../assets/stage-families.svg)

Um stage marcado com um ponto âmbar carrega estado de um frame para o
seguinte. Vale a pena saber disso por duas razões: é a única coisa que um
stage tem permissão de guardar, e é o que [o editor](gui.md) precisa
resetar em vez de realimentar quando você ajusta um frame pausado.

## Sources

| | |
|---|---|
| `video(path)` | frames de um arquivo de vídeo, em ordem |
| `image(path, fps)` | uma única imagem estática, como sequência de um frame |
| `folder(path, pattern, fps)` | toda imagem em um diretório, ordenada — um vídeo que não dá para avançar/retroceder |

As três também respondem a `count` e `read(index)`, o que é o que torna
possível buscar (seek) no editor. `VideoSource.read` rastreia sua própria
posição, então tocar para frente nunca toca em `CAP_PROP_POS_FRAMES` —
definir isso força uma busca por keyframe e uma redecodificação, muitas vezes
mais lenta que ler o próximo frame.

## Stages

**Ajuste** — `brightness_contrast(brightness, contrast)`, `saturation(gain)`,
`gamma(value)`, `gray`, `colorspace(to)`, `resize(size)`, `select(input)`

**Blur / morfologia** — `median_blur(ksize)`, `gaussian_blur(ksize, sigma)`,
`clahe(clip_limit, tile_grid)`, `morphology(op, ksize, iterations)`

**Threshold** — `threshold(value, mode, otsu)` — modos `binary`, `binary_inv`,
`trunc`, `tozero`, `tozero_inv` — `adaptive_threshold(method, block, c, invert)`

**Região de interesse** — `roi(x, y, w, h)`, `paste_roi(border, draw_on)`. O
recorte roda *antes* dos operadores que o leem, não depois: Otsu tira seu
nível de qualquer histograma que receber, e um blur lê vizinhos através da
borda, então analisar o frame inteiro e recortar no final deixaria pixels de
fora mudar números de dentro. `paste_roi` recoloca a região e desloca cada
coordenada em `ctx.rows` de volta para o espaço do frame.

**Bordas** — `canny(lo, hi)`, `sobel(ksize, dx, dy)`, `laplacian(ksize)`

**Geometria** — `hough_lines(...)`, `hough_circles(...)`, `harris(k, quality, max)`,
`shi_tomasi(max, quality, min_dist)`, `contours(min_area, mode, boxes, draw_on)` —
modos `external`, `list`, `tree` — `bounding_boxes(...)`,
`blobs(min_area, max_area, circularity, convexity, dark)`

**Keypoints** — `keypoints(detector, max, sensitivity, octaves, edge, rich)`, onde
`detector` é `sift` ou `orb`. `sensitivity` é normalizado 0..1 e mapeado para o
limiar nativo de cada detector, porque SIFT quer ~0.04 onde ORB quer ~20, e
nenhum config deveria precisar saber disso.

**Textura / cor** — `hog(orientations, cell, block)`, `lbp(points, radius, method)`,
`histogram(space, replace)`. HOG custa 150-300 ms em um frame 640x512 — tranquilo
para um lote, lento o bastante no editor para você sentir.

**Máscara** — `static_mask(threshold, invert)`, `apply_mask(input, mask, fill)`,
`mean_background(n_frames, use_mask, buffer)`, `color_select(space, ch0, ch1, ch2, fill)`

**Movimento** — `mog2(...)`, `knn(...)`, `frame_diff(lag)`, `three_frame_diff()`,
`farneback(pyr_scale, levels, winsize, iterations, gain)`,
`lucas_kanade(max_points, win, gain)`. Cada um deles emite a mesma coisa: uma
imagem de calor 0..255 de canal único. O que vem depois são stages comuns —

```yaml
- {type: farneback}
- {type: threshold, value: 25, mode: binary}
- {type: morphology, op: open, ksize: 3}
- {type: motion_objects, min_area: 50}
```

— então um threshold significa a mesma coisa qualquer que seja o algoritmo
acima dele, e trocar de algoritmo é uma linha. Depois `motion_heat(window)`
acumula, `heatmap(opacity, threshold, draw_on)` pinta, e
`motion_objects(min_area, max_travel, boxes, labels)` mede.

> `gain` nos stages de fluxo é uma **constante de calibração, não derivada**.
> No padrão de 32, 8 px/frame lê como escala cheia; uma pluma se movendo a
> menos de 1 px/frame precisa de vários múltiplos disso, ou uma abertura
> morfológica vai apagá-la. `configs/motion.yaml` usa 160 exatamente por esse
> motivo.

## Sinks

| | |
|---|---|
| `display(window, size, delay, quit_key)` | uma janela OpenCV; a tecla de saída interrompe a execução |
| `ffmpeg(path, fps, input)` | codifica para mp4 por um pipe libx264 |
| `image(path, input)` | um arquivo por frame — `{index}` no caminho numera uma sequência |
| `csv(dir, kinds)` | um CSV por tipo de linha, com uma coluna `frame` no início |
| `json(path)` | um objeto JSON de `ctx.metrics` por frame (JSON Lines) |
| `crops(dir, kind, input, pad)` | recorta o retângulo de cada linha de um frame e o salva |

`input:` nos sinks que produzem imagem escolhe o que eles escrevem; veja
[Escolhendo o que um sink produz](configuration.md#escolhendo-o-que-um-sink-produz).

## Adicionando um

Registre-o e ele passa a existir em todo lugar — nos configs, nas mensagens
de erro, na paleta do editor, com um formulário gerado:

```python
@register("stage", "my_stage")
class MyStage:
    def __init__(self, radius: int = 3):
        self.radius = radius

    def apply(self, ctx: Ctx) -> None:
        ctx.image = something(ctx.image, self.radius)
```

Duas coisas que o resto do sistema lê dessa classe sem que ninguém precise
avisar: `inspect.signature` fornece o formulário de parâmetros, e manter
`radius` como atributo é o que o marca como um botão *ao vivo* que o editor
pode atribuir enquanto pausado. Um parâmetro consumido na construção —
entregue a um objeto OpenCV e esquecido — é classificado como precisando de
reconstrução, o que está correto, porque é exatamente disso que ele precisa.
