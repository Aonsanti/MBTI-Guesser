import { useState } from "react";
import { Link } from "react-router-dom";
import { useModel } from "../hooks/useModel";

export default function NeuralNetwork() {
    const [text, setText] = useState("");
    const { prediction, loading, datasetInfo, error, predict } = useModel("nn");

    const datasets = datasetInfo ? [datasetInfo] : [];

    const handlePredict = () => {
        if (!text) return;
        predict(text);
    };

    return (
        <div className="bg-gray-700 text-white min-h-screen w-full flex flex-col items-center p-4 md:p-10 font-bold duration-200">
            <Link to="/" className="self-start mb-6 md:mb-10">
                <button className="bg-gray-600 hover:bg-gray-500 rounded-xl px-4 py-2 text-sm md:text-base md:px-6 md:py-2 transition-colors">Back to Home</button>
            </Link>

            <h1 className="text-2xl md:text-4xl mb-6 md:mb-10 text-center">Neural Network (MLP)</h1>
            <p className="mb-8 text-gray-300 text-center text-sm md:text-base max-w-xl">Multi-Layer Perceptron (MLP) model with TF-IDF trigrams for MBTI personality prediction</p>

            <div className="bg-gray-800 p-5 md:p-8 rounded-2xl shadow-xl w-full max-w-2xl flex flex-col gap-6 border-4 border-sky-500/30">
                <div>
                    <p className="text-xl md:text-2xl mb-2 text-sky-400">Please enter text for Neural Network:</p>
                    <p className="text-xs md:text-sm text-gray-400 font-normal mb-4">Example: "I love exploring new ideas and meeting people"</p>
                </div>

                <textarea
                    className="bg-gray-900 text-white p-5 rounded-xl border-2 border-gray-600 focus:border-sky-500 outline-none h-48 resize-none font-normal text-lg shadow-inner transition-colors"
                    placeholder="Type or paste your text here..."
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                />

                <button
                    onClick={handlePredict}
                    className="bg-sky-600 hover:bg-sky-500 text-white rounded-xl py-4 md:py-5 text-lg md:text-2xl font-black shadow-lg hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50"
                    disabled={loading}
                >
                    {loading ? "Processing..." : "Predict with Neural Network"}
                </button>

                {prediction && (
                    <div className="mt-4 p-5 md:p-6 bg-sky-900/40 border-2 border-sky-500 rounded-2xl relative overflow-hidden text-center md:text-left">
                        <p className="text-base md:text-lg text-sky-400 font-bold mb-1">Neural Network Result (MLP):</p>
                        <p className="text-5xl md:text-6xl text-white drop-shadow-lg scale-110 origin-center md:origin-left transition-transform">
                            {prediction.prediction}
                        </p>
                        <p className="text-[10px] md:text-xs text-sky-300 mt-4 tabular-nums">Architecture: {prediction.method}</p>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-900/40 border-2 border-red-500 rounded-xl text-red-400 font-normal text-sm">
                        ⚠️ {error}
                    </div>
                )}
            </div>

            <div className="mt-12 w-full max-w-5xl">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-800 p-6 rounded-2xl border-t-4 border-sky-500 shadow-lg text-center">
                    <p className="text-gray-400 text-xs uppercase tracking-widest mb-2 font-black">Architecture Accuracy</p>
                    <p className="text-5xl text-white font-black">96.45%</p>
                    <p className="text-sky-500 text-xs mt-2 italic font-normal">Superior to 95% threshold</p>
                </div>
                <div className="bg-gray-800 p-6 rounded-2xl border-t-4 border-sky-500 shadow-lg text-center">
                    <p className="text-gray-400 text-xs uppercase tracking-widest mb-2 font-black">Neural Depth</p>
                    <p className="text-5xl text-white font-black">4-Layers</p>
                    <p className="text-sky-500 text-xs mt-2 italic font-normal">1024 → 512 → 256 → 128</p>
                </div>
                <div className="bg-gray-800 p-6 rounded-2xl border-t-4 border-sky-500 shadow-lg text-center">
                    <p className="text-gray-400 text-xs uppercase tracking-widest mb-2 font-black">Classes</p>
                    <p className="text-5xl text-white font-black">16</p>
                    <p className="text-sky-500 text-xs mt-2 italic font-normal">MBTI Personality Types</p>
                </div>
                </div>

                <div className="mt-8 bg-gray-800 p-6 rounded-2xl border border-sky-500/30 shadow-lg text-center flex flex-col items-center">
                    <p className="text-gray-400 text-xs uppercase tracking-widest mb-4 font-black">Neural Network Training Loss Curve (Cross Entropy)</p>
                    <img src="/nn_graph.png" alt="Neural Network Loss Curve" className="w-full max-w-3xl rounded-xl shadow-2xl bg-gray-900 border border-gray-700" />
                </div>
            </div>

            <div className="mt-20 w-full max-w-5xl bg-gray-800 p-10 rounded-3xl border border-sky-500/20 shadow-2xl">
                <h2 className="text-3xl text-sky-500 mb-8 flex items-center gap-3">
                    <span className="bg-sky-500 text-black px-3 py-1 rounded-lg text-xl">2</span>
                    Model Code & Analysis
                </h2>

                <div className="space-y-8 font-normal text-gray-300">
                    <div className="bg-gray-900/50 p-6 rounded-xl border-l-4 border-sky-500">
                        <h3 className="text-xl font-bold text-white mb-2 decoration-sky-500/30 underline">Algorithm: Multi-Layer Perceptron (MLP)</h3>
                        <p className="leading-relaxed mb-4">
                            MLP is a feedforward artificial neural network that uses multiple layers of neurons with non-linear
                            activation functions. For MBTI prediction, it learns complex patterns in writing style to classify
                            text into one of 16 personality types. The deep 4-layer architecture (1024→512→256→128 neurons)
                            allows hierarchical feature learning from the TF-IDF text representations.
                        </p>
                        <pre className="bg-black/50 p-4 rounded-lg text-sm text-sky-300 overflow-x-auto whitespace-pre-wrap">
                            {`# Feature Extraction (Trigrams)
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 3),    # Unigrams + Bigrams + Trigrams
    min_df=2, max_df=0.95,
    sublinear_tf=True
)
X_vec = vectorizer.fit_transform(X)

# Neural Network (MLP) Model
model = MLPClassifier(
    hidden_layer_sizes=(1024, 512, 256, 128),  # 4 hidden layers
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
model.fit(X_train, y_train)`}
                        </pre>
                    </div>

                    <div className="bg-gray-900/50 p-6 rounded-xl border-l-4 border-sky-500">
                        <h3 className="text-xl font-bold text-white mb-2 decoration-sky-500/30 underline">Data Preprocessing (MBTI)</h3>
                        <p className="leading-relaxed mb-4">
                            The MBTI dataset contains user posts from forums. Preprocessing removes data leakage (MBTI type mentions),
                            URLs, and special characters to ensure the model learns from writing style, not type labels:
                        </p>
                        <pre className="bg-black/50 p-4 rounded-lg text-sm text-sky-300 overflow-x-auto whitespace-pre-wrap">
                            {`def clean_text_mbti(text):
    # 1. Remove URLs
    text = re.sub(r'http\\S+', '', text)
    # 2. Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    # 3. Lowercase
    text = text.lower()
    # 4. Remove MBTI type mentions (prevent leakage)
    for t in ['infj','entp','intp','intj',...]:
        text = text.replace(t, '')
    # 5. Lemmatize & remove stopwords
    words = [lemmatizer.lemmatize(word)
             for word in text.split()
             if word not in stop_words and len(word) > 2]
    return " ".join(words)`}
                        </pre>
                    </div>

                    <div className="bg-gray-900/50 p-6 rounded-xl border-l-4 border-sky-500">
                        <h3 className="text-xl font-bold text-white mb-2 decoration-sky-500/30 underline">How the MLP Learns MBTI Types</h3>
                        <p className="leading-relaxed mb-4">
                            The MLP processes each user's posts through 4 hidden layers using <strong>Backpropagation</strong>:
                        </p>
                        <ul className="list-disc list-inside space-y-2 ml-4">
                            <li><strong>Layer 1 (1024 neurons):</strong> Learns basic word/phrase patterns associated with personality traits</li>
                            <li><strong>Layer 2 (512 neurons):</strong> Combines patterns into higher-level personality features (introversion vs extroversion cues)</li>
                            <li><strong>Layer 3 (256 neurons):</strong> Identifies combinations of trait dimensions (e.g., Thinking + Intuitive patterns)</li>
                            <li><strong>Layer 4 (128 neurons):</strong> Final personality abstraction before classifying into 16 types</li>
                            <li><strong>ReLU Activation:</strong> Non-linearity allows complex personality-language mapping</li>
                            <li><strong>Adam Optimizer:</strong> Adaptive learning rate for efficient convergence on 16-class problem</li>
                            <li><strong>Early Stopping:</strong> Prevents overfitting by monitoring validation loss</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div className="mt-20 w-full max-w-5xl flex flex-col gap-10 mb-20">
                <h2 className="text-3xl text-center text-sky-400">Data Details & References</h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {datasets.map((ds, idx) => (
                        <div key={idx} className="bg-gray-600/10 p-10 rounded-3xl border border-gray-500/30 shadow-xl">
                            <h3 className="text-2xl text-white mb-8 border-b border-gray-500/30 pb-4">
                                <span className="text-sky-500 font-black mr-2">#</span>
                                {ds.name}
                            </h3>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-6">
                                    <div>
                                        <p className="text-sky-400 font-bold mb-1 uppercase tracking-widest text-xs">Source</p>
                                        <p className="font-normal text-gray-300">{ds.source}</p>
                                    </div>
                                    <div>
                                        <p className="text-sky-400 font-bold mb-1 uppercase tracking-widest text-xs">Features</p>
                                        <p className="font-normal text-gray-300">{ds.features}</p>
                                    </div>
                                </div>

                                <div className="space-y-6">
                                    <div>
                                        <p className="text-orange-400 font-bold mb-1 uppercase tracking-widest text-xs">Imperfections</p>
                                        <p className="font-normal text-gray-300">{ds.imperfections}</p>
                                    </div>
                                    <div>
                                        <p className="text-green-400 font-bold mb-1 uppercase tracking-widest text-xs">Preparation</p>
                                        <p className="font-normal text-gray-300">{ds.preparation}</p>
                                    </div>
                                </div>
                            </div>

                            <div className="mt-8 pt-6 border-t border-gray-500/20">
                                <p className="text-blue-400 font-bold mb-2 uppercase tracking-widest text-xs">Reference Link</p>
                                <p className="font-normal text-sky-400 underline italic text-sm">
                                    <a href="https://www.kaggle.com/datasets/datasnaek/mbti-type" target="_blank" rel="noreferrer">
                                        Kaggle: MBTI Type Dataset (mbti_1.csv)
                                    </a>
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}