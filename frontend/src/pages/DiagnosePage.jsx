import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { motion } from 'framer-motion';
import UploadCard from '../components/UploadCard';
import ParticleShaderField from '../components/ParticleShaderField';
import FallingLeaves from '../components/FallingLeaves';
import usePredictAPI from '../hooks/usePredictAPI';
import LeafBurstEffect from '../components/LeafBurstEffect';

const DiagnosePage = () => {
    const navigate = useNavigate();
    const { predict, loading } = usePredictAPI();

    const [image, setImage] = useState(null);
    const [question, setQuestion] = useState("");

    const handleFileSelect = (file) => {
        setImage(file);
    };

    const handleDiagnose = async () => {
        if (!image || !question) {
            alert("Please provide both an image and a question.");
            return;
        }

        console.log("Submitting prediction request...");
        const result = await predict(image, question);
        console.log("Result received:", result);

        if (result && result.trim().length > 0) {
            console.log("Navigating to result with:", result);
            navigate('/result', { state: { result, image: URL.createObjectURL(image) } });
        } else {
            console.warn("Received empty result from backend:", result);
            alert("The spirits are silent... (Backend returned empty result). check logs.");
        }
    };

    return (
        <div className="relative w-full h-screen bg-forest-900 overflow-hidden text-forest-100 font-sans">
            {/* 3D Background */}
            <div className="absolute inset-0 z-0 opacity-40">
                <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                    <ambientLight intensity={0.6} />
                    <ParticleShaderField count={100} color="#16a34a" />
                </Canvas>
                <FallingLeaves />
            </div>

            <div className="relative z-10 container mx-auto px-4 h-full flex flex-col items-center justify-center">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="w-full max-w-2xl bg-white/80 backdrop-blur-md border border-forest-800 p-8 rounded-2xl shadow-xl"
                >
                    <h2 className="text-3xl font-magic text-center mb-8 text-forest-700">Botanical Enquiry</h2>

                    <div className="mb-8">
                        <UploadCard onFileSelect={handleFileSelect} />
                    </div>

                    <div className="mb-8">
                        <label className="block text-sm uppercase tracking-wider mb-2 text-forest-700">Your Question</label>
                        <input
                            type="text"
                            className="w-full bg-white border border-forest-800 rounded-lg p-4 text-forest-100 focus:outline-none focus:border-magic-glow transition-colors placeholder-forest-100/50"
                            placeholder="e.g. Is this leaf healthy?"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                        />
                    </div>

                    <div className="flex justify-center">
                        <motion.button
                            onClick={handleDiagnose}
                            disabled={loading}
                            className={`
                                relative px-8 py-3 rounded-full font-bold uppercase tracking-widest text-sm
                                ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-magic-glow text-white hover:bg-forest-700'}
                                transition-all duration-300 shadow-md
                            `}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                        >
                            {loading ? (
                                <span className="flex items-center gap-2">
                                    <span className="animate-spin text-xl">✽</span> Divining...
                                </span>
                            ) : (
                                "Reveal Truth"
                            )}
                        </motion.button>
                    </div>
                </motion.div>
            </div>

            {/* Loading Overlay */}
            {loading && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="absolute inset-0 z-50 bg-white/80 backdrop-blur-sm flex items-center justify-center flex-col"
                >
                    <div className="w-24 h-24 mb-4">
                        <Canvas>
                            <ambientLight />
                            <pointLight position={[10, 10, 10]} />
                            <LeafBurstEffect active={true} />
                        </Canvas>
                    </div>
                    <p className="text-forest-700 font-magic text-xl animate-pulse">Nature is thinking...</p>
                </motion.div>
            )}
        </div>
    );
};

export default DiagnosePage;
