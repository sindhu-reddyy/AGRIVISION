import React, { useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload } from 'lucide-react';

const UploadCard = ({ onFileSelect }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [preview, setPreview] = useState(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setIsDragging(true);
        } else if (e.type === "dragleave") {
            setIsDragging(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    }, []);

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    };

    const handleFile = (file) => {
        setPreview(URL.createObjectURL(file));
        onFileSelect(file);
    };

    return (
        <motion.div
            className={`
                relative w-full max-w-md mx-auto h-64 border-2 border-dashed rounded-xl 
                flex flex-col items-center justify-center cursor-pointer overflow-hidden
                transition-colors duration-300
                ${isDragging ? 'border-magic-glow bg-magic-glow/10' : 'border-forest-100/30 bg-forest-900/40'}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            whileHover={{ scale: 1.02, boxShadow: "0 0 20px rgba(168, 255, 170, 0.2)" }}
        >
            <input
                type="file"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                onChange={handleChange}
                accept="image/*"
            />

            {preview ? (
                <img src={preview} alt="Preview" className="w-full h-full object-cover opacity-80" />
            ) : (
                <div className="flex flex-col items-center text-forest-100/70 z-0">
                    <Upload size={48} className="mb-4" />
                    <p className="text-sm font-magic tracking-wider">Drop mystical leaf here</p>
                </div>
            )}

            {/* Sparkles overlay could go here */}
        </motion.div>
    );
};

export default UploadCard;
