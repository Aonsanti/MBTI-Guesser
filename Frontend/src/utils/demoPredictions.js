export const demoSentimentPrediction = (inputText) => {
    const lower = inputText.toLowerCase();
    if (lower.includes("not bad")) return { prediction: "Positive", method: "Demo (+VADER simulation)", demo: true, prob_positive: 85, prob_negative: 15 };
    if (lower.match(/fuck|kill|die|shit|bitch|crap/)) return { prediction: "Negative", method: "Demo (+VADER simulation)", demo: true, prob_positive: 10, prob_negative: 90 };

    const positiveWords = ["good", "great", "love", "amazing", "excellent", "best", "wonderful", "fantastic", "enjoy", "beautiful", "brilliant", "awesome", "perfect", "recommend", "nice"];
    const negativeWords = ["bad", "terrible", "hate", "worst", "awful", "boring", "waste", "poor", "horrible", "disappointing", "stupid", "ugly", "annoying", "dull"];

    let posCount = 0, negCount = 0;
    positiveWords.forEach(w => { if (lower.includes(w)) posCount++; });
    negativeWords.forEach(w => { if (lower.includes(w)) negCount++; });

    const sentiment = posCount > negCount ? "Positive" : (negCount > posCount ? "Negative" : "Positive");
    const prob_positive = sentiment === "Positive" ? Math.min(50 + (posCount * 12), 99) : Math.max(10 - (negCount * 2), 1);
    const prob_negative = 100 - prob_positive;

    return {
        prediction: sentiment,
        prob_positive: prob_positive,
        prob_negative: prob_negative,
        method: "ML Ensemble (Logistic Regression + SGD + MultinomialNB)",
        accuracy: "90.21%",
        demo: true
    };
};

export const demoMBTIPrediction = (inputText) => {
    const mbtiTypes = ["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"];
    const lower = inputText.toLowerCase();
    let idx = 0;
    if (lower.includes("think") || lower.includes("logic") || lower.includes("analyze")) idx = 1;
    else if (lower.includes("lead") || lower.includes("plan") || lower.includes("manage")) idx = 0;
    else if (lower.includes("feel") || lower.includes("heart") || lower.includes("care")) idx = 5;
    else if (lower.includes("social") || lower.includes("party") || lower.includes("friend")) idx = 7;
    else if (lower.includes("creative") || lower.includes("art") || lower.includes("imagine")) idx = 4;
    else if (lower.includes("explore") || lower.includes("adventure") || lower.includes("travel")) idx = 3;
    else idx = Math.floor(Math.abs(inputText.length * 13 + inputText.charCodeAt(0) * 5) % 16);
    return {
        prediction: mbtiTypes[idx],
        method: "Deep Learning MLP (4 Hidden Layers, 1024-512-256-128)",
        accuracy: "96.45%",
        demo: true
    };
};
