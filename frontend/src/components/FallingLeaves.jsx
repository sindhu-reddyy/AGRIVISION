import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const NUM_LEAVES = 30;

const FallingLeaves = () => {
    const [leaves, setLeaves] = useState([]);

    useEffect(() => {
        const generatedLeaves = Array.from({ length: NUM_LEAVES }).map((_, i) => ({
            id: i,
            x: Math.random() * 100, // Random horizontal position %
            delay: Math.random() * 5, // Random start delay
            duration: 10 + Math.random() * 10, // Random fall duration (slow)
            rotation: Math.random() * 360, // Random initial rotation
            scale: 0.5 + Math.random() * 0.5, // Random size
        }));
        setLeaves(generatedLeaves);
    }, []);

    return (
        <div className="absolute inset-0 pointer-events-none overflow-hidden select-none">
            {leaves.map((leaf) => (
                <motion.div
                    key={leaf.id}
                    initial={{
                        y: -100,
                        x: `${leaf.x}vw`,
                        rotate: leaf.rotation,
                        opacity: 0,
                    }}
                    animate={{
                        y: '110vh',
                        x: `${leaf.x + (Math.random() * 20 - 10)}vw`, // Swaying effect
                        rotate: leaf.rotation + 360 + Math.random() * 180,
                        opacity: [0, 1, 1, 0], // Fade in then out at bottom
                    }}
                    transition={{
                        duration: leaf.duration,
                        ease: "linear",
                        repeat: Infinity,
                        delay: leaf.delay,
                    }}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        scale: leaf.scale,
                    }}
                >
                    {/* SVG Leaf Icon */}
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12.0001 22C12.0001 22 5.00012 18 5.00012 11C5.00012 7.13401 8.13413 4 12.0001 4C15.8661 4 19.0001 7.13401 19.0001 11C19.0001 18 12.0001 22 12.0001 22Z" fill="#22c55e" fillOpacity="0.4" stroke="#16a34a" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
                        <path d="M12 4V14" stroke="#16a34a" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </motion.div>
            ))}
        </div>
    );
};

export default FallingLeaves;
