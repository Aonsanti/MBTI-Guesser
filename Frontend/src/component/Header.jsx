export default function Header() {
    return (
        <>
            <header className="shadow-lg shadow-black relative z-10">
                <div className="flex items-center bg-[#15202b] text-white p-5">
                    <h1 className="flex-1 text-center text-[24px] md:text-[3vw] duration-300 font-bold">
                        Machine Learning & Neural Network
                    </h1>
                    <a href="https://github.com/Aonsanti/MBTI-Guesser" target="_blank" className="hover:opacity-75 transition-opacity">
                        <img src="/github-white-icon.webp" alt="GitHub" width="48" height="48" />
                    </a>
                </div>
            </header>
        </>
    )
}