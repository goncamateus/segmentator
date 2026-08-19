# Instalação

## A partir do código-fonte (para desenvolver, ou adicionar um stage)

Requer [uv](https://docs.astral.sh/uv/) e Python 3.13+.

```bash
git clone https://github.com/goncamateus/segmentator.git
cd segmentator
uv sync                 # apenas a CLI
uv run segmentator configs/baseline.yaml
```

O sink `ffmpeg` chama o binário `ffmpeg` externamente — instale-o separadamente
(`apt install ffmpeg`, `brew install ffmpeg`) e garanta que esteja no `PATH`. O
sink `display` e tudo o mais em `segmentator/ops/` não precisam de nada além
das dependências Python que o `uv sync` já instalou.

O editor fica atrás de um extra, então uma instalação headless — uma máquina de
lote, CI — nunca puxa o PyQt6:

```bash
uv sync --extra gui
uv run segmentator-gui configs/motion.yaml
```

Rode a suíte de testes com:

```bash
uv run pytest
```

## App pré-compilado (só o editor)

Cada [release marcada](https://github.com/goncamateus/segmentator/releases)
publica um build autônomo do editor para Linux e macOS — sem Python, sem `uv`,
sem dependências para instalar. Ele empacota o mesmo código de
`segmentator-gui`; a CLI em si continua sendo executada a partir do código-fonte
(acima), já que uma ferramenta de lote não tem uso para um app com duplo clique.

**Linux — AppImage.**

```bash
chmod +x segmentator-*.AppImage
./segmentator-*.AppImage configs/motion.yaml
```

Sem etapa de instalação; ele roda no lugar onde está.

**macOS — dmg.**

Abra o `.dmg`, arraste `Segmentator.app` para `Applications`. O build não é
notarizado, então o Gatekeeper bloqueia o primeiro lançamento — clique com o
botão direito no app e escolha *Abrir*, depois confirme.

**Windows.** Sem instalador pré-compilado. Compile a partir do código-fonte
(acima) e depois empacote você mesmo com
`uv run --group build pyinstaller --noconfirm segmentator.spec` — veja
[segmentator.spec](https://github.com/goncamateus/segmentator/blob/main/segmentator.spec)
e os scripts em `packaging/` para como Linux e macOS fazem isso.
