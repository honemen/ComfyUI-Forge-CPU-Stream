# Forge CPU Stream — 研究版ノード

Forge NeoとComfyUIの観測差を調べるため、ER SDEの狭い条件とtoken報告を公開する実験用ノードです。全面的な互換レイヤーではありません。通常利用には[一般版](../Forge%20CPU%20Stream一般版/README_ja.md)を推奨します。別途使用する測定器は[research資料](../research/README_ja.md)を参照してください。

## インストールとworkflow

`forge_cpu_stream/`と`forge_conditioning_minimal/`をそれぞれComfyUIの`custom_nodes/`へコピーし、付属ファイルを同じフォルダに保持して再起動します。通常のエンコードにグローバルなmonkeypatchは不要です。利用中のComfyUI sampler APIとPyTorchを使います。JSON内の`silvermoonmix_v60VPred.safetensors`は同梱されていません。

[`Forge CPU Stream　研究版.json`](Forge%20CPU%20Stream%E3%80%80研究版.json)では、checkpoint→`ModelSamplingDiscrete(v_prediction, zsnr=true)`→`CFGGuider`と`BetaSamplingScheduler(36, 0.6, 0.6)`、正負の**Forge CPU Stream Text Encode (Minimal)**→guider、`KSamplerSelect(er_sde)`→**Forge CPU Stream Adapter**→`SamplerCustomAdvanced`の`noise`と`sampler`に**対の出力**を接続しています。`EmptyLatentImage(1024,1024,batch 1)`もAdvanced samplerへつなぎ、その第1出力→`VAEDecode`→`PreviewImage`です。保存済みCFGは4、Text Encode modeは`chunk_and_original`、adapterの`initial_scale`は`forge_off`です。checkpointとseedは必要に応じて変更してください。PNGは参照スクリーンショットです。

## ノードと制約

| ノード | 入出力 | コード上の動作 |
| --- | --- | --- |
| **Forge CPU Stream Adapter** | `SAMPLER`、64 bit符号なし`seed`、`initial_scale` (`comfy_auto`, `forge_off`, `forge_on`) → `NOISE`, `SAMPLER` | 標準の`KSamplerSelect(er_sde)`だけを受け付けます。CPU generatorで初期ノイズを作り、同じgeneratorをER SDEの追加ノイズに使います。2つの出力は1回限りの受け渡しとして連動します。scaleの3モードはそれぞれComfyUIに委ねる、`max_denoise=False`固定、`True`固定です。 |
| **Forge CPU Stream Text Encode (Minimal)** | `CLIP`、複数行の`text`、`mode` → `CONDITIONING`、JSON文字列`token_report` | SDXL L/Gのchunkごとのtoken IDと重みを報告します。下記2モードを持ちます。 |

Adapterは`[1,4,H,W]`のfloat32、ゼロ初期化、非nestedの`EmptyLatentImage`を要求し、indexed batch、noise mask、既存noise上書き、seed不一致、shape/dtype変更、最後がゼロでない不完全なsigma列、最大sigma以外からの開始を拒否します。同じ`SamplerCustomAdvanced`に対の出力をつないでください。一般版と異なり任意sampler、部分denoise、画像からのlatentは対象外です。これらは実行時チェックであり、将来のComfyUI版の動作保証ではありません。

`chunk_only`はForge由来のchunk構築後、token重みをComfyUI encoderに渡します。`chunk_and_original`は重みを1にしてencoderに渡し、出力conditioningの77 token chunkごとにL/Gへ別々のOriginal強調を適用します。通常のSDXL CLIP L/G、L/Gで同じchunk数、77 token tokenizerが必要です。Forge式LoRAタグ、textual inversion構文、`AND`合成は拒否されます。LoRAは通常のComfyUIノードで読み込んでください。どちらのモードもForge Neoとの出力一致を証明するものではありません。

## Level 3 traceとの接続

Minimalノードは`FC_TRACE3_DIR`が設定された場合、`fc_trace3`をimportします。[`../research/level3/fc_trace3.py`](../research/level3/fc_trace3.py)と依存する`fc_trace.py`をComfyUIのPython環境からimport可能にしてください。環境変数だけではインストールされません。L/Gのtoken化を記録し、`chunk_and_original`では`adapter_result`のcontext/pooled sessionも記録します。`FC_TRACE3_TARGET_JSON`には対象文字列を収めたJSONファイルを指定できます。アプリ側の追加観測には別途Level 3 patchが必要です。適用前に互換性を確認してください。traceが無効ならtrace用import経路は実行されません。

## 検証範囲と協力

深く調査したのは**ER SDEだけ**で、他samplerの同等性は未検証です。主な実測環境は**silvermoonmix v60 (V-Pred) + ER SDE**です。checkpoint専用の作りではありませんが、他checkpointの体系的な検証はありません。CPU noise streamは観測差の一部を減らすものであり、最終画像の完全一致は確立していません。conditioningの比較には未解決の差があり、根本原因の全面的な特定を意味しません。観測値と範囲はresearchのJSONを参照してください。

独立検証、修正、別環境、より良い測定器を歓迎します。同等以上の既存ノード、より広範な実装、ComfyUI ↔ Forge Neoの相互互換実装があればIssueで共有してください。帰属・ライセンスの確認事項は一般版とresearchのREADMEを参照してください。ZIPにLICENSE/NOTICEはありません。
