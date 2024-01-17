# LocalGPT: Sichere, lokale Konversationen mit Ihren Dokumenten 🌐

**LocalGPT** ist eine Open-Source-Initiative, die es Ihnen ermöglicht, mit Ihren Dokumenten zu kommunizieren, ohne Ihre Privatsphäre zu gefährden. Da alles lokal ausgeführt wird, können Sie sicher sein, dass keine Daten jemals Ihren Computer verlassen. Tauchen Sie ein in die Welt der sicheren, lokalen Dokumenteninteraktionen mit LocalGPT.

## Funktionen 🌟
- **Höchste Privatsphäre**: Ihre Daten verbleiben auf Ihrem Computer und gewährleisten 100% Sicherheit.
- **Vielfältige Modellunterstützung**: Integrieren Sie nahtlos verschiedene Open-Source-Modelle, einschließlich HF, GPTQ, GGML und GGUF.
- **Vielfältige Einbettungen**: Wählen Sie aus einer Reihe von Open-Source-Einbettungen.
- **Wiederverwendung Ihres LLM**: Sobald heruntergeladen, können Sie Ihr LLM ohne wiederholte Downloads wiederverwenden.
- **Chatverlauf**: Merkt sich Ihre vorherigen Konversationen (in einer Sitzung).
- **API**: LocalGPT verfügt über eine API, die Sie für den Aufbau von RAG-Anwendungen verwenden können.
- **Grafische Benutzeroberfläche**: LocalGPT wird mit zwei GUIs geliefert, eines verwendet die API und das andere ist eigenständig (basierend auf Streamlit).
- **GPU-, CPU- und MPS-Unterstützung**: Unterstützt mehrere Plattformen von Haus aus. Chatten Sie mit Ihren Daten unter Verwendung von `CUDA`, `CPU` oder `MPS` und mehr!

## Tauchen Sie tiefer ein mit unseren Videos 🎥
- [Detaillierte Code-Durchlauf](https://youtu.be/MlyoObdIHyo)
- [Llama-2 mit LocalGPT](https://youtu.be/lbFmceo4D5E)
- [Hinzufügen von Chatverlauf](https://youtu.be/d7otIM_MCZs)
- [LocalGPT - Aktualisiert (17.09.2023)](https://youtu.be/G_prHSKX9d4)

## Technische Details 🛠️
Durch die Auswahl der richtigen lokalen Modelle und die Verwendung von `LangChain` können Sie die gesamte RAG-Pipeline lokal ausführen, ohne dass Daten Ihre Umgebung verlassen, und dies mit vernünftiger Leistung.

- `ingest.py` verwendet `LangChain`-Tools, um das Dokument zu analysieren und Einbettungen lokal mit `InstructorEmbeddings` zu erstellen. Anschließend wird das Ergebnis in einer lokalen Vektordatenbank mit `Chroma`-Vektorspeicher gespeichert.
- `run_localGPT.py` verwendet ein lokales LLM, um Fragen zu verstehen und Antworten zu erstellen. Der Kontext für die Antworten wird aus der lokalen Vektordatenbank extrahiert, indem eine Ähnlichkeitssuche durchgeführt wird, um das richtige Stück Kontext aus den Dokumenten zu finden.
- Sie können dieses lokale LLM durch ein beliebiges anderes LLM von HuggingFace ersetzen. Stellen Sie sicher, dass das von Ihnen ausgewählte LLM im HF-Format vorliegt.

Dieses Projekt wurde von dem ursprünglichen [privateGPT](https://github.com/imartinez/privateGPT) inspiriert.

## Erstellt mit 🧩
- [LangChain](https://github.com/hwchase17/langchain)
- [HuggingFace LLMs](https://huggingface.co/models)
- [InstructorEmbeddings](https://instructor-embedding.github.io/)
- [LLAMACPP](https://github.com/abetlen/llama-cpp-python)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

# Umgebung einrichten 🌍

1. 📥 Klone das Repository mit Git:

```shell
git clone https://github.com/PromtEngineer/localGPT.git
```

2. 🐍 Installiere [conda](https://www.anaconda.com/download) für das Verwalten von virtuellen Umgebungen. Erstelle und aktiviere eine neue virtuelle Umgebung.

```shell
conda create -n localGPT python=3.10.0
conda activate localGPT
```

3. 🛠️ Installiere die Abhängigkeiten mit pip

Um deine Umgebung einzurichten und den Code auszuführen, installiere zunächst alle Anforderungen:

```shell
pip install -r requirements.txt
```

***Installation von LLAMA-CPP:***

LocalGPT verwendet [LlamaCpp-Python](https://github.com/abetlen/llama-cpp-python) für GGML (du benötigst llama-cpp-python <=0.1.76) und GGUF (llama-cpp-python >=0.1.83) Modelle.


Wenn du BLAS oder Metal mit [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal) verwenden möchtest, kannst du entsprechende Flags setzen:

Für `NVIDIA`-GPUs-Unterstützung, verwende `cuBLAS`

```shell
# Beispiel: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Für Apple Metal (`M1/M2`) Unterstützung, verwende

```shell
# Beispiel: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

Für weitere Details siehe bitte [llama-cpp](https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal).

## Docker 🐳

Die Installation der erforderlichen Pakete für die GPU-Inferenz auf NVIDIA-GPUs, wie gcc 11 und CUDA 11, kann Konflikte mit anderen Paketen auf deinem System verursachen. Als Alternative zu Conda kannst du Docker mit dem bereitgestellten Dockerfile verwenden. Es enthält CUDA, dein System benötigt nur Docker, BuildKit, deinen NVIDIA-GPU-Treiber und das NVIDIA-Container-Toolkit. Erstelle es mit `docker build -t localgpt .`, erfordert BuildKit. Docker BuildKit unterstützt derzeit keine GPU während der *docker build*-Zeit, nur während der *docker run*-Zeit. Starte es mit `docker run -it --mount src="$HOME/.cache",target=/root/.cache,type

=bind --gpus=all localgpt`.

## Testdatensatz

Für Tests enthält dieses Repository die [Verfassung der USA](https://constitutioncenter.org/media/files/constitution.pdf) als Beispieldatei.

## Eigenen Daten einfügen.
Lege deine Dateien in den Ordner `SOURCE_DOCUMENTS`. Du kannst mehrere Ordner innerhalb des Ordners `SOURCE_DOCUMENTS` erstellen, und der Code liest deine Dateien rekursiv.

### Unterstützte Dateiformate:
LocalGPT unterstützt derzeit die folgenden Dateiformate. LocalGPT verwendet `LangChain` zum Laden dieser Dateiformate. Der Code in `constants.py` verwendet ein `DOCUMENT_MAP`-Wörterbuch, um ein Dateiformat auf den entsprechenden Loader in [LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/) abzubilden.

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Einlesen

Führe den folgenden Befehl aus, um alle Daten einzulesen.

Wenn du `cuda` auf deinem System eingerichtet hast:

```shell
python ingest.py
```
Du siehst eine Ausgabe wie diese:
<img width="1110" alt="Screenshot 2023-09-14 at 3 36 27 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/c9274e9a-842c-49b9-8d95-606c3d80011f">


Verwende das Argument für den Gerätetyp, um ein bestimmtes Gerät anzugeben.
Um auf `cpu` auszuführen:

```sh
python ingest.py --device_type cpu
```

Um auf `M1/M2` auszuführen:

```sh
python ingest.py --device_type mps
```

Verwende die Hilfe für eine vollständige Liste der unterstützten Geräte.

```sh
python ingest.py --help
```

Dies erstellt einen neuen Ordner namens `DB` und verwendet ihn für den neu erstellten Vektorspeicher. Du kannst beliebig viele Dokumente einlesen, und alle werden in der lokalen Einbettungsdatenbank akkumuliert. Wenn du von einer leeren Datenbank aus starten möchtest, lösche die `DB` und führe deine Dokumente erneut ein.

Hinweis: Wenn du dies zum ersten Mal ausführst, benötigt es Internetzugang, um das Einbettungsmodell herunterzuladen (Standard: `Instructor Embedding`). Bei den folgenden Ausführungen verlässt keine Daten deine lokale Umgebung, und du kannst Daten ohne Internetverbindung einlesen.

## Stelle Fragen zu deinen Dokumenten, lokal!

Um mit deinen Dokumenten zu chatten, führe den folgenden Befehl aus (standardmäßig wird es auf `cuda` ausgeführt).

```shell
python run_localGPT.py
```
Du kannst auch den Gerätetyp angeben, genau wie bei `ingest.py`

```shell
python run_localGPT.py --device_type mps # zum Ausführen auf Apple Silicon
```

Dies lädt den eingelesenen Vektorspeicher und das Einbettungsmodell. Du erhältst eine Eingabeaufforderung:

```shell
> Gib eine Anfrage ein:
```

Nachdem du deine Frage eingegeben hast, drücke Enter. LocalGPT benötigt je nach Hardware einige Zeit. Du erhältst eine Antwort wie diese unten.
<img width="1312" alt="Screenshot 2023-09-14 at 3 33 19 PM" src="https://github.com/PromtEngineer/localGPT/assets/134474669/a7268de9-ade0-420b-a00b-ed12207dbe41">

Sobald die Antwort generiert ist, kannst du eine weitere Frage stellen, ohne das Skript erneut auszuführen. Warte einfach auf die erneute Eingabeaufforderung.

***Hinweis:*** Wenn du dies zum ersten Mal ausführst, benötigt es eine Internetverbindung, um das LLM herunterzuladen (Standard: `TheBloke/Llama-2-7b-Chat-GGUF`). Danach kannst du deine Internetverbindung ausschalten, und die Skript-Inferenz wird immer noch funktionieren. Keine Daten verlassen deine lokale Umgebung.

Gib `exit` ein, um das Skript zu beenden.

### Zusätzliche Optionen mit run_localGPT.py

Du kannst die `--show_sources`-Flagge mit `run_localGPT.py` verwenden, um anzuzeigen, welche Chunks vom Einbettungsmodell abgerufen wurden. Standardmäßig werden 4 verschiedene Quellen/Chunks angezeigt. Du kannst die Anzahl der Quellen/Chunks ändern.

```shell
python run_localGPT.py --show_sources
```

Eine andere Option ist das Aktivieren des Chatverlaufs. ***Hinweis***: Dies ist standardmäßig deaktiviert und kann mit der `--use_history`-Flagge aktiviert werden. Das Kontextfenster ist begrenzt, also beachte, dass die Aktivierung des Verlaufs es verwenden wird und überlaufen kann.

```shell
python run_localGPT.py --use_history
```

# Führe die grafische Benutzeroberfläche aus

1. Öffne `constants.py` in einem Editor deiner Wahl und füge je nach Auswahl das LLM hinzu, das du verwenden möchtest. Standardmäßig wird das folgende Modell verwendet:

```shell
MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
```

3. Öffne ein Terminal und aktiviere deine Python-Umgebung, die die aus requirements.txt installierten Abhängigkeiten enthält.

4. Navigiere zum Verzeichnis `/LOCALGPT`.

5. Führe den folgenden Befehl aus: `python run_localGPT_API.py`. Die API sollte gestartet werden.

6. Warte, bis alles geladen ist. Du solltest etwas wie `INFO:werkzeug:Press CTRL+C to quit` sehen.

7. Öffne ein zweites Terminal und aktiviere dieselbe Python-Umgebung.

8. Navigiere zum Verzeichnis `/LOCALGPT/localGPTUI`.

9. Führe den Befehl `python localGPTUI.py` aus.

10. Öffne einen Webbrowser und gehe zur Adresse `http://localhost:5111/`.

# Wie wähle ich verschiedene LLM-Modelle aus?

Um die Modelle zu ändern, musst du sowohl `MODEL_ID` als auch `MODEL_BASENAME` festlegen.

1. Öffne `constants.py` in deinem Editor.

2. Ändere `MODEL_ID` und `MODEL_BASENAME`. Wenn du ein quantisiertes Modell (`GGML`, `GPTQ`, `GGUF`) verwendest, musst du `MODEL_BASENAME` angeben. Für unquantisierte Modelle setze `MODEL_BASENAME` auf `NONE`.

5. Es gibt eine Vielzahl von Beispielmodellen von HuggingFace, die bereits getestet wurden, um mit dem ursprünglichen trainierten Modell verwendet zu werden (endet mit HF oder hat eine .bin in seinen "Files and versions"), und quantisierten Modellen (endet mit GPTQ oder hat eine .no-act-order oder .safetensors in seinen "Files and versions").
6. Für Modelle, die mit HF enden oder eine .bin-Datei in ihrer "Files and versions" auf ihrer HuggingFace-Seite haben.

   - Stelle sicher, dass du ein `MODEL_ID` ausgewählt hast. Zum Beispiel: `MODEL_ID = "TheBloke/guanaco-7B-HF"`
   - Gehe zum [HuggingFace Repo](https://huggingface.co/TheBloke/guanaco-7B-HF).

7. Für Modelle, die GPTQ in ihrem Namen haben oder eine .no-act-order oder .safetensors-Erweiterung in ihren "Files and versions" auf ihrer HuggingFace-Seite haben.

   - Stelle sicher, dass du ein `MODEL_ID` ausgewählt hast. Zum Beispiel: `MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"`
   - Gehe zum entsprechenden [HuggingFace Repo](https://huggingface.co/TheBloke/wizardLM-7B-GPTQ) und wähle "Files and versions" aus.
   - Wähle einen der Modellnamen aus und setze ihn als `MODEL_BASENAME`. Zum Beispiel: `MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"`

8. Folge denselben Schritten für GGUF- und GGML-Modelle.

# GPU- und VRAM-Anforderungen

Im Folgenden sind die VRAM-Anforderungen für verschiedene Modelle je nach ihrer Größe (Milliarden von Parametern) aufgeführt. Die Schätzungen in der Tabelle enthalten nicht den VRAM, der von den Einbettungsmodellen verwendet wird - diese verwenden zusätzlich 2 GB bis 7 GB VRAM, abhängig vom Modell.

| Modellgröße (Mrd.) | float32   | float16   | GPTQ 8-Bit      | GPTQ 4-Bit          |
| ------- | --------- | --------- | -------------- | ------------------ |
| 7 Mrd.  | 28 GB     | 14 GB     | 7 GB - 9 GB    | 3,5 GB - 5 GB      |
| 13 Mrd. | 52 GB     | 26 GB     | 13 GB - 15 GB  | 6,5 GB - 8 GB      |
| 32 Mrd. | 130 GB    | 65 GB     | 32,5 GB - 35 GB| 16,25 GB - 19 GB   |
| 65 Mrd. | 260,8 GB  | 130,4 GB  | 65,2 GB - 67 GB| 32,6 GB - 35 GB    |

# Systemanforderungen

## Python-Version

Um diese Software zu verwenden, musst du Python 3.10 oder eine neuere Version installiert haben. Frühere Versionen von Python werden nicht kompiliert.

## C++-Compiler

Wenn du während des `pip install`-Prozesses einen Fehler beim Erstellen eines Wheels erhältst, musst du möglicherweise einen C++-Compiler auf deinem Computer installieren.

### Für Windows 10/11

Um einen C++-Compiler unter Windows 10/11 zu installieren, befolge diese Schritte:

1. Installiere Visual Studio 2022.
2. Stelle sicher, dass die folgenden Komponenten ausgewählt sind:
   - Universal Windows Platform Development
   - C++ CMake Tools für Windows
3. Lade den MinGW-Installer von der [MinGW-Website](https://sourceforge.net/projects/mingw/) herunter.
4. Führe den Installer aus und wähle die "gcc"-Komponente aus.

### Probleme mit den NVIDIA-Treibern:

Folge dieser [Seite](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04), um die NVIDIA-Treiber zu installieren.

## Sterne-Historie

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)

# Haftungsausschluss

Dies ist ein Testprojekt, um die Machbarkeit einer vollständig lokalen Lösung für die Beantwortung von Fragen unter Verwendung von LLMs und Vektoreinbettungen zu validieren. Es ist nicht für den Einsatz in der Produktion geeignet und soll nicht in der Produktion verwendet werden. Vicuna-7B basiert auf dem Llama-Modell und unterliegt der ursprünglichen Llama-Lizenz.

# Häufige Fehler

- [Torch nicht mit aktiviertem CUDA kompatibel](https://github.com/pytorch/py

torch/issues/30664)

  - CUDA-Version abrufen
    ```shell
    nvcc --version
    ```
    ```shell
    nvidia-smi
    ```
  - Versuche, PyTorch je nach deiner CUDA-Version zu installieren
    ```shell
       conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
    ```
  - Wenn es nicht funktioniert, versuche es erneut zu installieren
    ```shell
       pip uninstall torch
       pip cache purge
       pip install torch -f https://download.pytorch.org/whl/torch_stable.html
    ```

- [ERROR: Der Abhängigkeitslöser von pip berücksichtigt derzeit nicht alle installierten Pakete](https://stackoverflow.com/questions/72672196/error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-pa/76604141#76604141)
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```

- [Fehler beim Importieren von transformers](https://github.com/huggingface/transformers/issues/11262)
  - Versuche die Neuinstallation
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```
