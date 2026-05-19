document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const text = document.getElementById('newsText').value;
    const resultBox = document.getElementById('resultBox');
    const resultText = document.getElementById('resultText');
    const confidenceText = document.getElementById('confidenceText');
    const btn = document.getElementById('analyzeBtn');

    if (!text.trim()) return;

    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: text,
                source_id: 100 // Default to Unknown for manual paste
            })
        });

        const data = await response.json();

        resultBox.style.display = 'block';
        resultBox.className = `result ${data.prediction.toLowerCase()}`;
        resultText.innerText = data.prediction === "FAKE" ? "🚩 FAKE NEWS DETECTED" : "✅ VERIFIED REAL";
        confidenceText.innerText = `Confidence: ${(data.confidence_score * 100).toFixed(2)}% | AI Intelligence Active`;

    } catch (error) {
        console.error(error);
        alert("Bhai, server check karo! FastAPI running hai?");
    } finally {
        btn.innerText = "Analyze with SOTA AI";
        btn.disabled = false;
    }
});

// Auto-fill from selection if available
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    chrome.scripting.executeScript({
        target: {tabId: tabs[0].id},
        function: () => window.getSelection().toString()
    }, (results) => {
        if (results && results[0].result) {
            document.getElementById('newsText').value = results[0].result;
        }
    });
});
