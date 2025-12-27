const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("result");
const logsTable = document.getElementById("logsTable");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert numeric fields to numbers
    Object.keys(data).forEach(key => {
        if (!isNaN(data[key]) && data[key] !== '') data[key] = Number(data[key]);
    });

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const result = await response.json();

        resultDiv.innerHTML = `
            Prediction: <b>${result.prediction}</b><br>
            Probability: <b>${result.probability}</b>
        `;

        loadLogs(); // refresh logs table
    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerHTML = `<span style="color:red;">Error: ${error.message}</span>`;
    }
});

async function loadLogs() {
    try {
        const response = await fetch("/logs");
        if (!response.ok) throw new Error(`Server returned ${response.status}`);

        const logs = await response.json();
        logsTable.innerHTML = "";

        logs.forEach(log => {
            logsTable.innerHTML += `
                <tr>
                    <td>${log.full_name || "-"}</td>
                    <td>${log.timestamp}</td>
                    <td>${log.loan_amount}</td>
                    <td>${log.annual_income}</td>
                    <td>${log.credit_score}</td>
                    <td>${log.loan_term}</td>
                    <td>${log.prediction}</td>
                    <td>${log.probability}</td>
                </tr>
            `;
        });
    } catch (error) {
        console.error("Error loading logs:", error);
        logsTable.innerHTML = `<tr><td colspan="8" style="color:red;">Error loading logs</td></tr>`;
    }
}

// initial load
loadLogs();
