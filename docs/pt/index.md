# Segmentator

Segmentação de imagem e vídeo orientada por configuração. Um pipeline é descrito em
YAML, então testar uma cadeia de pré-processamento, detector ou modelo de fundo
diferente é uma edição de config, não uma edição de código.

```bash
uv run segmentator configs/baseline.yaml
uv run segmentator configs/structure.yaml
uv run segmentator configs/mog2_contours.yaml --video inputs/gasvid1.mp4 --max-frames 300
```

`--video` e `--output` sobrescrevem o caminho da fonte e o caminho do primeiro
sink ffmpeg para uma execução.

Também há um editor, que constrói esses configs com uma pré-visualização ao vivo
de cada stage e de cada sink:

```bash
uv sync --extra gui
uv run segmentator-gui configs/motion.yaml
```

Parte da suíte [goncanalyser](https://github.com/goncamateus/goncanalyser).
goncanalyser é o workspace Qt onde você *ajusta* uma cadeia de operadores à mão;
segmentator é o motor headless que *executa* a receita ajustada sobre um lote.
Todo operador que o goncanalyser expõe existe aqui como um stage, e os dois
concordam numericamente no mesmo frame com os mesmos parâmetros.

## Para onde ir agora

| | |
|---|---|
| [Instalação](installation.md) | `uv sync`, o extra `gui`, e o AppImage / dmg pré-compilados |
| [Escrevendo um config](configuration.md) | o formato de config: `Ctx`, taps, `input:`, ramificação com `select` |
| [O catálogo](stages.md) | toda source, stage e sink, e o que cada parâmetro significa |
| [O editor](gui.md) | o editor, e cada operação que ele permite |

## Arquitetura

Um pipeline é `source → stages → sinks`. Stages são uma lista ordenada aplicada a
cada frame; nada ramifica com base em estado de execução, então não há máquina de
estados aqui — o único "estado" é aquecimento-de-fundo vs. estável, que
`BackgroundModel.ready` trata com um `if`.

```python
@dataclass
class Ctx:
    image: np.ndarray   # working array; stages rebind it
    source: np.ndarray  # original BGR frame, untouched
    index: int
    store: dict         # side channel within one frame
    taps: dict          # named stages' outputs, for sinks to pick from
    metrics: dict       # per-frame scalars, what the json sink writes
    rows: dict          # per-object detail, what the csv and crops sinks write

class Stage(Protocol):
    def apply(self, ctx: Ctx) -> None: ...

class Sink(Protocol):
    def write(self, ctx: Ctx) -> bool: ...   # False stops the run
    def close(self) -> None: ...
```

Componentes se registram por nome (`@register("stage", "median_blur")`), e
`build()` transforma um mapeamento `{type: ..., **params}` em uma instância. Um
nome desconhecido lança um erro com a lista dos nomes conhecidos. Nada mais
precisa ser dito ao sistema: o formato do config, as mensagens de erro e a
paleta e os formulários do editor são todos lidos a partir do registro e dos
construtores.

| Arquivo | Conteúdo |
|---|---|
| [cli.py](https://github.com/goncamateus/segmentator/blob/main/cli.py) | CLI |
| [segmentator/pipeline.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/pipeline.py) | `Ctx`, protocolos, registro, `Pipeline` |
| [segmentator/io.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/io.py) | sources e sinks |
| [segmentator/ops/](https://github.com/goncamateus/segmentator/tree/main/segmentator/ops) | operadores puros, sem protocolo de stage, sem objeto de config |
| [segmentator/stages/](https://github.com/goncamateus/segmentator/tree/main/segmentator/stages) | os stages registrados, por família |
| [segmentator/background_model.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/background_model.py) | fundo fixo de média de N frames |
| [segmentator/video_writer.py](https://github.com/goncamateus/segmentator/blob/main/segmentator/video_writer.py) | pipe ffmpeg/libx264 |
| [segmentator/gui/](https://github.com/goncamateus/segmentator/tree/main/segmentator/gui) | o editor PyQt6 — opcional, `--extra gui` |

## Paridade com o goncanalyser

Todo operador é portado de `features/*` do goncanalyser — os corpos das funções,
não uma reimplementação — com cada campo de `Settings` virando um argumento de
construtor no stage que o usa. No mesmo frame com os mesmos parâmetros os dois
produzem saída idêntica: `edge_px`, contagens de cantos e keypoints, vetores
HOG, códigos LBP, gráficos de histograma, linhas de contornos e de Hough — tudo
bate exatamente, apesar dos dois repositórios usarem builds diferentes do
OpenCV.

`tests/test_stages.py` guarda as próprias asserções `_demo()` do goncanalyser,
reapontadas para os stages — portadas, não inventadas, então uma suíte que
passa significa que os dois repositórios concordam nos números, não apenas que
os dois rodam.
