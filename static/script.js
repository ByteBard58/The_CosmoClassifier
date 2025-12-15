document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("predictForm");
    const resultDiv = document.getElementById("result");
    const predClassDiv = document.getElementById("predClass");
    const probsDiv = document.getElementById("probabilities");
    const submitBtn = document.getElementById("predictBtn");

    // Seamless navigation: Focus next input on specific keypresses if desired (optional)
    // Basic "Enter" key on forms naturally submits, which is standard.

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // UI Feedback: Loading State
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = "<span>Analysing Cosmos...</span>";
        submitBtn.disabled = true;
        resultDiv.classList.add("hidden");

        try {
            const formData = new FormData(form);
            const response = await fetch("/predict", { 
                method: "POST", 
                body: formData 
            });

            if (!response.ok) throw new Error("Prediction failed");

            const data = await response.json();
            
            // Render Prediction Result
            const pred = data.prediction;
            const probs = data.probabilities;

            const predLabelSpan = `<span class="pred-label">${pred}</span>`;
            predClassDiv.innerHTML = `Identified Object: ${predLabelSpan}`;

            // Render Probabilities
            probsDiv.innerHTML = "";
            
            // Allow animation frame to clear before rendering bars for transition effect
            for (const [cls, prob] of Object.entries(probs)) {
                // Determine width
                const percentage = (prob * 100).toFixed(1);
                
                probsDiv.innerHTML += `
                    <div class="prob-row">
                        <span class="prob-name">${cls}</span>
                        <div class="prob-bar-container">
                            <div class="prob-fill" style="width: 0%" data-width="${percentage}%"></div>
                        </div>
                        <span class="prob-value">${percentage}%</span>
                    </div>`;
            }

            // Show Results
            resultDiv.classList.remove("hidden");
            
            // Trigger animation for bars
            setTimeout(() => {
                const bars = document.querySelectorAll(".prob-fill");
                bars.forEach(bar => {
                    bar.style.width = bar.getAttribute("data-width");
                });
            }, 100);
            
            // Scroll to result
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } catch (error) {
            console.error(error);
            alert("An error occurred while communicating with the observatory.");
        } finally {
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
        }
    });
});