import { Link } from "react-router-dom";

export default function Home() {
    return (
        <div className="bg-gray-700 text-white min-h-screen w-full flex flex-col md:flex-row justify-center items-center gap-10 md:gap-40 font-bold p-10 duration-200">

            <Link to="/MachineLearning">
                <button className="hover:bg-green-500 rounded-xl w-40 h-20 md:w-60 md:h-32 text-sm md:text-xl hover:scale-110">
                    Machine Learning
                </button>
            </Link>

            <Link to="/NeuralNetwork">
                <button className="hover:bg-sky-500 rounded-xl w-40 h-20 md:w-60 md:h-32  text-sm md:text-xl hover:scale-110">
                    Neural Network
                </button>
            </Link>

        </div>
    );
}