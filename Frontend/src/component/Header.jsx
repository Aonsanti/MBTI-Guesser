export default function Header() {
    return (
        <>
            <header className="shadow-lg shadow-black relative z-10">
                <div className="flex items-center bg-[#15202b] text-white p-5">
                    <h1 className="flex-1 text-center text-xl md:text-3xl lg:text-4xl duration-300 font-bold px-4">
                        Machine Learning & Neural Network
                    </h1>
                    <a href="https://github.com/Aonsanti/MBTI-and-Sentiment-Prediction" target="_blank" rel="noopener noreferrer">
                        <img src="/github-white-icon.webp" alt="GitHub" className="w-8 h-8 md:w-12 md:h-12" />
                    </a>
                </div>
            </header>
        </>
    )
}