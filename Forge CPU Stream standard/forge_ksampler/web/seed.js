import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "forge_ksampler.manual_seed",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ForgeKSampler") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            this.addWidget("button", "Random seed", null, () => {
                const seed = this.widgets?.find(w => w.name === "seed");
                if (!seed || this.inputs?.some(i => i.name === "seed" && i.link != null)) return;
                const words = new Uint32Array(2);
                let next;
                do {
                    crypto.getRandomValues(words);
                    next = (words[0] & 0x1fffff) * 4294967296 + words[1];
                } while (next === seed.value);
                seed.value = next;
                seed.callback?.(next);
                this.setDirtyCanvas(true, true);
                app.graph?.change?.();
            }, { serialize: false });
            return result;
        };
    },
});
