import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const CinematicTransition = ({ onComplete }) => {
    return (
        <AnimatePresence onExitComplete={onComplete}>
            <motion.div
                className="fixed inset-0 z-[100] bg-forest-900 flex items-center justify-center pointer-events-none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1 }}
            >
                <div className="absolute inset-0 bg-black/40" />
                <motion.div
                    className="w-full h-full absolute top-0 left-0 bg-forest-800"
                    initial={{ scaleX: 0, originX: 0 }}
                    animate={{ scaleX: 1 }}
                    exit={{ scaleX: 0, originX: 1 }}
                    transition={{ duration: 0.8, ease: "easeInOut" }}
                />
            </motion.div>
        </AnimatePresence>
    );
};

export default CinematicTransition;
