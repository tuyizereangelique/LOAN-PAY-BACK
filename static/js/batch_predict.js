const batchFileInput = document.getElementById("batchFile");
const batchPredictBtn = document.getElementById("batchPredictBtn");
const resultDiv = document.getElementById("batchResult");

batchPredictBtn.addEventListener("click", async () => {
    resultDiv.innerHTML = "";

    // 1️⃣ Validate file
    if (!batchFileInput.files.length) {
        resultDiv.innerHTML = `
            <div class="alert warning">
                 <b>No file selected</b><br>
                Please upload a valid CSV file to continue.
            </div>
        `;
        return;
    }

    const formData = new FormData();
    formData.append("file", batchFileInput.files[0]);

    try {
        // 2️⃣ Loading message
        resultDiv.innerHTML = `
            <div class="alert info">
                ⏳ Processing batch prediction, please wait...
            </div>
        `;

        // 3️⃣ Send to backend
        const response = await fetch("/batch_predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Batch prediction failed");
        }

        let results = await response.json();

        // ✅ Ensure results is an array
        results = Array.isArray(results) ? results : Object.values(results);

        // 4️⃣ Statistics
        const total = results.length;
        const willRepay = results.filter(r => r.prediction === "Will Repay").length;
        const willNotRepay = total - willRepay;

        // 5️⃣ Professional summary message
        resultDiv.innerHTML = `
            <div class="alert success">
                 <b>Batch Prediction Completed Successfully</b><br><br>
            </div>
        `;
        
        await loadLogs();



    } catch (error) {
        resultDiv.innerHTML = `
            <div class="alert error">
                 <b>Batch Prediction Failed</b><br>
                ${error.message}
            </div>
        `;
    }
});
