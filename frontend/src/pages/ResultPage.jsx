import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Canvas } from '@react-three/fiber';
import { motion } from 'framer-motion';
import ParticleShaderField from '../components/ParticleShaderField';
import FallingLeaves from '../components/FallingLeaves';

const ResultPage = () => {
    const { state } = useLocation();
    const navigate = useNavigate();

    // Redirect if no state
    useEffect(() => {
        if (!state) navigate('/diagnose');
    }, [state, navigate]);

    if (!state) return null;

    return (
        <div className="relative w-full h-screen bg-forest-900 overflow-hidden text-forest-100">
            {/* 3D Background */}
            <div className="absolute inset-0 z-0">
                <Canvas camera={{ position: [0, 0, 10], fov: 60 }}>
                    <ambientLight intensity={0.6} />
                    <ParticleShaderField count={300} color="#22c55e" />
                </Canvas>
                <FallingLeaves />
            </div>

            <div className="relative z-10 container mx-auto px-4 h-full flex items-center justify-center">
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ type: "spring", duration: 1 }}
                    className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8 bg-white/90 backdrop-blur-xl border border-forest-800 p-8 rounded-3xl shadow-2xl"
                >
                    <div className="relative h-64 md:h-auto rounded-xl overflow-hidden border border-forest-800 group shadow-inner">
                        <img src={state.image} alt="Analyzed Leaf" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
                    </div>

                    <div className="flex flex-col justify-center">
                        <h2 className="text-sm font-bold text-forest-700 uppercase tracking-widest mb-2">Analysis Complete</h2>
                        <h1 className="text-4xl font-magic text-forest-100 mb-6 drop-shadow-sm">The Garden Speaks...</h1>

                        <div className="bg-forest-900 border-l-4 border-magic-glow p-6 rounded-r-xl mb-8 shadow-sm">
                            <p className="text-lg leading-relaxed text-forest-100 font-medium">
                                {state.result || "The spirits are murmuring unintelligibly... (No result text provided)"}
                            </p>
                        </div>

                        <button
                            onClick={() => navigate('/diagnose')}
                            className="self-start px-6 py-3 border-2 border-forest-100/20 hover:bg-forest-100 hover:text-white rounded-lg transition-colors flex items-center gap-2 group text-forest-700 font-bold"
                        >
                            <span>← Consult Again</span>
                        </button>
                    </div>
                </motion.div>
            </div>
        </div>
    );
};

export default ResultPage;
