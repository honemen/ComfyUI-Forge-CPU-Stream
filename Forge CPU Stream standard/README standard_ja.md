# Forge CPU Stream — 一般利用向け

Forge NeoとComfyUIの間で観測されたサンプリング・テキストエンコードの実装差の一部を減らす、実験的なComfyUIノードです。最終画像の完全一致や全面的な互換性は保証しません。差分調査には[研究版](../Forge%20CPU%20Stream研究版/README_ja.md)と[research資料](../research/README_ja.md)を参照してください。

## インストール

`forge_ksampler/`と`forge_text_encode/`の**フォルダをそれぞれ**ComfyUIの`custom_nodes/`にコピーし、Pythonファイルと`forge_ksampler/web/`の構成を保持してComfyUIを再起動します。通常のノード利用にはresearchのpatchやForge本体は不要です。利用中のComfyUIとPyTorch、および読み込んだCLIPのtokenizerを使います。対象ComfyUIの固定バージョンや自動インストーラは付属せず、上流のsampler API変更時にはエラーになり得ます。

## ノード

| ノード | 入力 | 出力と役割 |
| --- | --- | --- |
| **Forge CPU Stream KSampler** | `model`、64 bit符号なし`seed`、`steps`、`cfg`、`sampler_name`、`scheduler`、`beta_alpha`、`beta_beta`、正負の`CONDITIONING`、`latent_image`、`denoise` | `LATENT`を1つ出力します。スケジュール（`beta`時はalpha/beta指定）、CFG guider、samplerを組み立て、初期ノイズをCPU generatorで生成します。対象の独立Gaussianノイズを使うsamplerでは同じgeneratorの乱数列を追加ノイズにも継続使用します。`denoise=0`では入力latentのコピーを返します。 |
| **Forge CPU Stream Text Encode** | `clip`、複数行の`text` | `CONDITIONING`を1つ出力します。Forge由来の強調構文とSDXL L/Gの77 token単位のchunkを使い、各encoder・各chunkにOriginal強調を適用します。 |

samplerの選択肢はインストール済みComfyUIの一覧に従います。CPU追加ノイズの上書き対象は`er_sde`, `euler_ancestral`, `euler_ancestral_cfg_pp`, `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `dpmpp_2s_ancestral_cfg_pp`, `ddpm`, `lcm`, `res_multistep`, `res_multistep_cfg_pp`, `res_multistep_ancestral`, `res_multistep_ancestral_cfg_pp`, `seeds_2`, `seeds_3`, `exp_heun_2_x0_sde`, `sa_solver`, `sa_solver_pece`です。それ以外の選択可能なsamplerでは初期CPUノイズを供給しますが、solver側のnoise samplerは上書きしません。Brownian tree型の相関ノイズは元の実装に委ねます。**深く調査したsamplerはER SDEだけです。他のsamplerは体系的な比較・検証を行っておらず、Forge Neoとの同等動作は保証しません。** `er_sde`については、この実装で`max_denoise=False`に固定します。

ブラウザ上の**Random seed**ボタンは未接続のseedを押したときだけ変更し、生成ごとの自動ランダム化は行いません。出力ラベルの変更は表示のみです。Text Encodeは通常のSDXL CLIP L/G、77 token chunkを要求し、Forge式`<lora:...>`タグ、`embedding:`、`AND`合成を拒否します。LoRAはComfyUIの通常のLoRAノードで読み込んでください。

## workflow例

[`Forge CPU Stream　一般版.json`](Forge%20CPU%20Stream%E3%80%80一般版.json)をComfyUIで開きます。`CheckpointLoaderSimple`→`ModelSamplingDiscrete(v_prediction, zsnr=true)`→KSampler、CLIP→正負それぞれのText Encode、`EmptyLatentImage`→KSampler、出力latent→`VAEDecode`→`PreviewImage`という構成です。保存値はER SDE、beta 0.6/0.6、36 steps、CFG 4、1024×1024、batch 1、`denoise=1`です。JSONが参照する`silvermoonmix_v60VPred.safetensors`は同梱されていません。手元のcheckpointと必要なprompt/seedに設定し直してください。PNGは説明用スクリーンショットです。

## 検証範囲と制限

主な実測環境は**silvermoonmix v60 (V-Pred) + ER SDE**です。ノードをこのcheckpoint専用にはしていませんが、他checkpointは体系的に検証していません。Forge NeoとComfyUIのnoise/RNG挙動を追跡し、CPU乱数列の継続使用を実装しても最終出力の完全一致には至っていません。その後conditioning/text encodingも調査し、一般版Text Encodeには研究版の`chunk_and_original`経路を反映しましたが、画像一致を示すものではありません。他のハードウェア、batch条件、mask、将来のComfyUI版について同等性は未確認です。samplerは既存のnoise上書きを拒否し、対象経路で`noise_sampler` APIを検査します。

追加検証、別環境での結果、修正、他samplerの調査、より良い実装を歓迎します。同等以上の既存ノード、より広範なForge Neo対応、ComfyUI ↔ Forge Neo間の相互互換実装をご存じならIssueで参照先を共有してください。関連実装として歓迎します。

## 公開前の帰属・ライセンス確認

一般版と研究版の`forge_chunks.py`には、[Forge Classic/Neoのcommit `0c9273f`のparser](https://github.com/Haoming02/sd-webui-forge-classic/blob/0c9273f69dbe491246fb11c3d4ac70123099d393/backend/text_processing/parsing.py)由来という記載があります。この帰属コメントを保持してください。添付ZIPには**LICENSE/NOTICEがありません**。派生部分とresearchのpatchを含め、正確な上流ライセンスと整合性を公開前に確認してください。適切なLICENSEと、参照元・commitを記すNOTICEが必要になる可能性があります。ここでパッケージのライセンスを断定しません。
