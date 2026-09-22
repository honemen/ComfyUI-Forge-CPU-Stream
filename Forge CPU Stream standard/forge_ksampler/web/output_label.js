import { app } from "../../scripts/app.js";

// Presentation only: keep the output's name, type, slot index and links intact.
function setOutputLabel(node) {
    const output = node.outputs?.[0];
    if (output?.type === "LATENT") {
        output.label = "LATENT";
    }
}

app.registerExtension({
    name: "forge_ksampler.output_label",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ForgeKSampler") return;
        for (const hook of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                setOutputLabel(this);
                return result;
            };
        }
    },
});