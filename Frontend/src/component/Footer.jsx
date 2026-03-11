export default function Footer() {
    return (
        <footer className="w-full bg-[#15202b] border-t border-white/5 py-8 mt-auto">
            <div className="container mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-4">
                
                <div className="text-center md:text-left">
                    <p className="text-white opacity-80 text-sm md:text-base">
                        Developed by <span className="font-bold text-green-400">Santi Ngamcharee</span>
                    </p>
                </div>

                <div className="text-center md:text-right">
                    <p className="text-gray-400 text-xs uppercase tracking-widest">
                        © 2026 Santi Ngamcharee. All Rights Reserved
                    </p>
                </div>

            </div>
        </footer>
    )
}