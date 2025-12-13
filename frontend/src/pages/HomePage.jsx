import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import CinematicTransition from '../components/CinematicTransition';
import FallingLeaves from '../components/FallingLeaves';


const HomePage = () => {
    const navigate = useNavigate();
    const [transitioning, setTransitioning] = useState(false);

    const handleEnter = () => {
        setTransitioning(true);
        setTimeout(() => {
            navigate('/diagnose');
        }, 1500); // Wait for transition
    };

    return (
        <div className="relative w-full h-screen bg-forest-900 overflow-hidden">
            {/* 2D Background & Animation */}
            <div className="absolute inset-0 z-0">
                <FallingLeaves />
            </div>

            {/* UI Overlay */}
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center pointer-events-none">
                <motion.h1
                    className="text-6xl md:text-8xl font-magic text-forest-100 text-center tracking-widest drop-shadow-lg mb-8"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 1.5 }}
                >
                    AgriVision
                </motion.h1>

                <motion.p
                    className="text-xl text-forest-700 font-light tracking-widest mb-12 uppercase bg-white/50 backdrop-blur-sm px-4 py-1 rounded"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5, duration: 1 }}
                >
                    Intelligence Rooted in Nature
                </motion.p>

                <motion.button
                    className="pointer-events-auto px-12 py-4 border-2 border-forest-700 text-forest-100 font-bold font-magic text-xl backdrop-blur-md bg-white/40 hover:bg-forest-700 hover:text-white transition-all duration-300 rounded-lg shadow-lg"
                    onClick={handleEnter}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                >
                    ENTER THE GARDEN
                </motion.button>
            </div>

            {/* Transition Overlay */}
            {transitioning && <CinematicTransition />}
        </div>
    );
};

export default HomePage;
