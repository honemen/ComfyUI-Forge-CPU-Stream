# research — 調査資料と測定器

Forge Neo ↔ ComfyUIの今回の調査を第三者が確認・継続できるよう残した資料です。完成した汎用デバッグツール集でも、普遍的に検証された測定法でもありません。深く調査したsamplerは**ER SDEだけ**です。他samplerは体系的に検証しておらず、同じ測定点が適切か、同等の情報が得られるかも保証できません。主な実測環境は**silvermoonmix v60 (V-Pred) + ER SDE**です。ノード自体はこのcheckpoint専用の設計ではありませんが、他checkpointは体系的に検証していません。noise/RNGの差を減らしても最終画像の完全一致には至らず、続けてconditioning/text encodingを調査しました。

資料は**Level 1–2（sampler/noise、CFG/UNet）→ Level 3（text/conditioning）→ conditioning investigation（個別の追加調査）**の順です。最初に観測した数値差を、すべての後続差の原因と断定しません。大容量の生traceログは同梱しておらず、要約JSONとソースを収録しています。再比較には対象ソースの版を確認した上で新たなtraceが必要です。

## `level1-2/`：サンプリングの観測

| ファイル | 実装から確認できる役割 |
| --- | --- |
| `fc_trace.py` | 共通logger。tensorをNumPyファイルにコピーし、`events.jsonl`、実行環境とソースhashを新しい`pass_NNN`に記録します。`FC_TRACE_DIR`で有効化、`FC_TRACE_LEVEL=1`でsolver、`2`でCFG/UNet packetも観測します。`FC_TRACE_STEPS`はLevel 2の対象step（初期値`0`、または`all`）です。コードの前提は1 process、単一GPU、batch 1、通常のtxt2imgです。 |
| `patches/forge_level1.patch`, `comfy_level1.patch` | 各アプリの初期noise、sigma列、ER SDE solver、stepごとのdenoised/latent、追加noiseに計測点を挿入します。 |
| `patches/forge_level2.patch`, `comfy_level2.patch` | CFG出力と条件付き/無条件UNet入出力を観測します。対応するLevel 1側のimportが前提です。 |
| `compare_traces.py` | 2つの`pass_NNN`をstep/name/occurrenceで対応させ、バイト完全一致、値・shape・dtype差、最初に**観測した**差をJSONへ出力します。NumPyが必要で、欠測は一致とみなしません。 |
| `probe_sigma_precision.py` | 合成したCPUスケジュールでdtype処理順の差を確認する補助実験です。PyTorch、NumPy、SciPyが必要です。実際のユーザー実行時sigmaが同じ差を示す証拠ではありません。 |
| `patches/baseline_normalized_sha256.json` | patch作成時のForge/ComfyUI対象ソースの正規化hashです。 |

再観測時は、両アプリで`fc_trace.py`をimport可能にし、**手元のソース**とpatch対象・baseline hashを照合してください。適合する研究patchだけを確認して使い、アプリごとに新しい`FC_TRACE_DIR`を指定して条件を揃えます。比較例：`python level1-2/compare_traces.py FORGE/pass_000 COMFY/pass_000 --output comparison.json`。traceの負荷や対象版の違いにも注意してください。

## `level3/`：text/conditioningの観測

| ファイル | 実装から確認できる役割 |
| --- | --- |
| `fc_trace3.py` | `fc_trace.py`の`Trace`に依存する追加logger。encode/chunk、CLIP、assembly、任意のLoRA情報を`session_NNNN`、`session.json`、`index.jsonl`とtensor eventに記録します。新しい`FC_TRACE3_DIR`で有効化し、`FC_TRACE3_LAYERS`、`FC_TRACE3_DETAIL_LAYERS`、`FC_TRACE3_WEIGHTS=1`でCLIP観測の詳細を指定します。想定範囲は通常のSDXL、encoder呼び出し当たり1 text、1 processです。 |
| `patches/forge_level3.patch`, `comfy_level3.patch` | それぞれのアプリにparser、token、CLIP各段、SDXL context/pooled/vector assemblyの計測点を追加します。ノードのインストール作業ではありません。 |
| `patches/forge_level3_lora.patch`, `comfy_level3_lora.patch` | 任意のLoRAロード情報観測用です。 |
| `compare_traces_v3.py`, `compare_conditioning.py` | 前者はNumPyによるsession tensor比較、後者はencode/sessionの対応付けと段階別比較です。後者は`--list ROOT`、一意の完全一致textとencoderによるroot間対応付け、またはsessionの直接指定を受け付けます。欠測を報告し、実行順からpositive/negativeを推定しません。 |
| `inspect_safetensors.py` | モデルtensorを読み込まず、標準ライブラリだけでsafetensorsのheaderとファイルSHA256を調べます。 |
| `observed_v3_summary.json`, `parser_source_probe.json` | 選択した実行の比較要約（noiseの比較行を含む）とparser入力例の結果です。全環境への一般化はできません。 |
| `verification.json`, `KIT_SHA256.json`, `patches/baseline_normalized_sha256.json` | 実施した検証と未実施の項目、キット内ファイルのhash、対象ソースのhashです。特に`verification.json`は、ユーザーのGPU/checkpoint/LoRA/promptでのLevel 3実行をその検証で実施したとはしていません。 |

`fc_trace3.py`と`fc_trace.py`をPythonからimport可能にしてください。比較例：`python level3/compare_conditioning.py FORGE_ROOT COMFY_ROOT --output comparison_level3.json`。trace出力先には未作成の新しいディレクトリが必要です。patchの適用可能性は対象アプリの実版ごとに確認してください。

## `conditioning_investigation/`：個別の追加調査

| ファイル | 実装から確認できる役割 |
| --- | --- |
| `evidence.json`, `validation.json` | 正負のL/G token・embedding・layerとLoRAの比較、chunk境界の再現、CPU上での強調演算の反実仮想比較と制限事項です。`validation.json`はCPU演算の結果がGPU上のビット完全一致の証明ではないと明記します。 |
| `forge_level3_hotfixed_for_reverse.patch` | SDXL assembly metadataのkeywordを`prompt_texts`に変えたForge Level 3 patchの派生版です。reverse/check調査用であり、すべての版に対する置き換えではありません。 |
| `chunk_boundary_reproducer.py` | Forgeのchunk境界とComfyの単一大token groupを比較する標準ライブラリのみの最小再現です。同梱の`forge_chunks.py`を使用してForge側のchunk構築を再現します。 |
| `optional_clip0_probe/clip0_probe.py`, `clip0_only.patch` | CLIP layer 0の演算operandを観測する任意のprobeと`fc_trace3.py`向けpatchです。`FC_CLIP0_PROBE`、`FC_TRACE3_TARGET_JSON`、`FC_CLIP0_SAVE_WEIGHTS`等を使い、eager dense CLIPの特定演算を対象とします。 |
| `optional_clip0_probe/compare_clip0.py`, `compare_traces_v3.py` | negative encodeの重点比較とpass比較器の複製です。NumPyが必要です。 |
| `optional_clip0_probe/baseline.json`, `negative_prompt.json` | 任意patch適用前のLevel 3 loggerの正規化hashと、対象prompt選択用のJSON文字列です。 |
| `SHA256.json` | 調査ファイルのhash一覧です。 |

任意probeはeffective weightを大きく出力する場合があります。これらのJSONは限定された条件下の観測であり、普遍的な原因や最終画像の同等性を確立しません。

## patch・帰属・継続調査

patchは**調査当時のソース配置向け**です。現在のComfyUI / Forge Neoへそのまま適用できる保証はありません。版とbaseline hashを確認し、`git apply --check`と差分の目視確認を行い、復元可能な作業ツリーで調査してください。測定器は今回のER SDE調査用instrumentationであり、公式の完成済み検証一式ではありません。網羅的・学術的に検証済みの測定法とも主張しません。独立検証、修正、測定点の改善、他samplerの調査を歓迎します。

一般版と研究版の`forge_chunks.py`には[Forge Classic/Neoのcommit `0c9273f`のparser](https://github.com/Haoming02/sd-webui-forge-classic/blob/0c9273f69dbe491246fb11c3d4ac70123099d393/backend/text_processing/parsing.py)への帰属があります。既存コメントを保持してください。上流Forge/ComfyUIのファイルに対するpatchを含み、セットには**LICENSE/NOTICEがありません**。公開前に参照元の正確な版と派生部分のライセンスを確認し、適切なLICENSE/NOTICEで出典と条件を示してください。baseline hashはライセンス情報ではありません。ここでパッケージのライセンスは断定しません。

同等以上の既存ノード、より広いForge Neo対応、ComfyUI ↔ Forge Neo間の相互互換実装をご存じならIssueで参照先を共有してください。別環境での実測、誤りの修正、方法の変更や拡張を歓迎します。
