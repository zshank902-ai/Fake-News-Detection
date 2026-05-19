console.log("🛡️ The Truth Shield: Infiltrating Social Media...");

const badgeStyle = `
  background: #6366f1;
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: bold;
  margin-left: 8px;
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  border: none;
`;

async function injectBadges() {
    // Target X (Twitter) tweets
    const tweets = document.querySelectorAll('[data-testid="tweetText"]:not(.shield-scanned)');
    
    for (let tweet of tweets) {
        tweet.classList.add('shield-scanned');
        const text = tweet.innerText;
        
        if (text.length < 20) continue;

        const badge = document.createElement('button');
        badge.innerText = "🛡️ Verify";
        badge.style = badgeStyle;
        
        badge.onclick = async (e) => {
            e.stopPropagation();
            badge.innerText = "⌛ Scanning...";
            
            try {
                const response = await fetch('http://localhost:8000/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text, source_id: 50 }) // 50 = Social Media
                });
                const data = await response.json();
                
                if (data.prediction === "FAKE") {
                    badge.innerText = "🚩 FAKE";
                    badge.style.background = "#ef4444";
                } else {
                    badge.innerText = "✅ REAL";
                    badge.style.background = "#22c55e";
                }
            } catch (err) {
                badge.innerText = "⚠️ Error";
                badge.style.background = "#666";
            }
        };

        // Find the parent to append the badge next to the username or timestamp if possible
        // For simplicity, we append to the tweet text block
        tweet.appendChild(badge);
    }
}

// Observe for dynamic loading
const observer = new MutationObserver(injectBadges);
observer.observe(document.body, { childList: true, subtree: true });

// Initial run
injectBadges();
