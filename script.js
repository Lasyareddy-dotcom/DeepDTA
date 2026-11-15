const SCALE_MIN = 5;
const SCALE_MAX = 11;

const LEVELS = {
    Strong:   { text: "Strong binding",   range: "pKd 7 and above",   cls: "strong" },
    Moderate: { text: "Moderate binding", range: "pKd 6 to under 7",  cls: "moderate" },
    Weak:     { text: "Weak binding",     range: "pKd 5 to under 6",  cls: "weak" },
};

function showScale(value, label) {
    const info = LEVELS[label] || LEVELS.Weak;
    const el = document.getElementById("categoryLabel");
    el.textContent = info.text + " (" + info.range + ")";
    el.className = "category-label " + info.cls;

    const clamped = Math.min(Math.max(value, SCALE_MIN), SCALE_MAX);
    const percent = (clamped - SCALE_MIN) / (SCALE_MAX - SCALE_MIN) * 100;
    document.getElementById("marker").style.left = percent + "%";
    document.getElementById("scaleSection").hidden = false;
}

function showMessages(messages, isError) {
    const box = document.getElementById("warnings");
    box.innerHTML = "";
    messages.forEach(function (text) {
        const div = document.createElement("div");
        div.className = isError ? "error-text" : "warn-box";
        div.textContent = text;
        box.appendChild(div);
    });
}

async function predictAffinity() {
    const smiles = document.getElementById("smiles").value.trim();
    const protein = document.getElementById("protein").value.trim();
    const result = document.getElementById("resultValue");
    const button = document.getElementById("predictButton");
    const buttonText = document.getElementById("buttonText");

    if (smiles === "") { alert("Please enter a Drug SMILES."); return; }
    if (protein === "") { alert("Please enter a Protein Sequence."); return; }

    result.textContent = "...";
    document.getElementById("scaleSection").hidden = true;
    showMessages([], false);
    button.disabled = true;
    buttonText.textContent = "Predicting...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ smiles: smiles, protein: protein }),
        });
        if (!response.ok) {
            result.textContent = "—";
            showMessages(["The server could not read that input. Check that both boxes are filled and not too long."], true);
            return;
        }
        const data = await response.json();
        if (data.error) {
            result.textContent = "—";
            showMessages([data.error], true);
            return;
        }
        result.textContent = data.value.toFixed(4);
        showScale(data.value, data.label);
        showMessages(data.warnings, false);
    } catch (error) {
        console.error(error);
        result.textContent = "—";
        showMessages(["Could not reach the server. If it was idle, wait a minute and try again."], true);
    } finally {
        button.disabled = false;
        buttonText.textContent = "Predict Binding Affinity";
    }
}
