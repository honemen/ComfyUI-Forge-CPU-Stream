# ComfyUI Forge CPU Stream

ComfyUIとForge Neoの間で確認された一部の実装差を調査し、その差を小さくすることを目的とした、実験的なComfyUIノードおよび研究資料です。

本プロジェクトは、ピクセル単位での完全再現や完全な互換性を主張するものではありません。

## 内容

### 一般利用向けノード

`Forge CPU Stream standard/` を参照してください。

通常利用向け：

- Forge CPU Stream KSampler
- Forge CPU Stream Text Encode
- サンプルworkflow

### 研究用ノード

`Forge CPU Stream research_nodes/` を参照してください。

調査中に使用した実験的なノード：

- Forge CPU Stream Adapter
- Forge CPU Stream Text Encode (Minimal)
- 研究用workflow

### 研究ツール・資料

`research/` を参照してください。

調査で使用したLevel 1–2、Level 3、およびconditioning調査用のツール・資料を収録しています。

## 検証範囲

本調査はER SDEを中心に行っています。

他のサンプラーについては体系的な調査・検証を行っていないため、同等の動作や再現性は保証できません。

主な検証環境では、silvermoonmix v60（V-Pred）とER SDEを使用しています。ノード自体をこのcheckpointのみに制限する意図はありませんが、他のcheckpointについては体系的な検証を行っていません。

## 今後の調査について

追加検証、修正、測定方法の改善、異なる環境での検証結果、およびさらなる互換性調査を歓迎します。

ComfyUIとForge Neoの間で、同等またはより完全な互換性、あるいは双方向の互換性を実現する既存のComfyUIノードや実装をご存じの場合は、Issueでお知らせください。

今回の調査で見落としている可能性のある関連実装についての情報も歓迎します。

## ライセンスと帰属

`LICENSE` および `NOTICE` を参照してください。
