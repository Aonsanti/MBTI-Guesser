import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import Header from './component/Header.jsx'
import Footer from './component/Footer.jsx'
import MachineLearning from './pages/MachineLearning.jsx'
import NeuralNetwork from './pages/NeuralNetwork.jsx'

export default function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/MachineLearning' element={<MachineLearning />} />
        <Route path='/NeuralNetwork' element={<NeuralNetwork />} />
      </Routes>
      <Footer />
    </Router>
  )
}
