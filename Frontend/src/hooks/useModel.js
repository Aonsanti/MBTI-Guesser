import { useState, useEffect } from "react";
import { demoSentimentPrediction, demoMBTIPrediction } from "../utils/demoPredictions";

const API_URL = import.meta.env.DEV ? "http://localhost:8000/api" : "/api";

export function useModel(type = "ml") {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [datasetInfo, setDatasetInfo] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch(API_URL + "/dataset-info")
            .then(res => res.json())
            .then(data => {
                setDatasetInfo(type === "ml" ? data.ml_dataset : data.nn_dataset);
            })
            .catch(() => {
                const fallback = type === "ml" ? {
                    name: "IMDB Dataset of 50K Movie Reviews",
                    source: "Kaggle",
                    features: "Movie review text",
                    imperfections: "HTML tags, special characters, etc.",
                    preparation: "Cleaning, TF-IDF, etc."
                } : {
                    name: "MBTI Myers-Briggs Type Indicator",
                    source: "Kaggle",
                    features: "User posts",
                    imperfections: "Data leakage, unstructured text, etc.",
                    preparation: "Cleaning, TF-IDF with trigrams."
                };
                setDatasetInfo(fallback);
            });
    }, [type]);

    const predict = async (text) => {
        if (!text) return;
        setLoading(true);
        setError(null);
        setPrediction(null);
        try {
            const res = await fetch(`${API_URL}/predict/${type}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            if (!res.ok) throw new Error("Server error");
            const data = await res.json();
            setPrediction(data);
        } catch (err) {
            console.error(err);
            const fallback = type === "ml" ? demoSentimentPrediction(text) : demoMBTIPrediction(text);
            setPrediction(fallback);
        } finally {
            setLoading(false);
        }
    };

    return { prediction, loading, datasetInfo, error, predict };
}
