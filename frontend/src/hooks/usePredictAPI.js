import { useState } from 'react';
import axios from 'axios';

const usePredictAPI = () => {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const predict = async (imageFile, question) => {
        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('question', question);

        try {
            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            setResult(response.data.answer);
            return response.data.answer;
        } catch (err) {
            setError(err.response?.data?.detail || "Something went wrong in the magical forest...");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return { predict, loading, result, error };
};

export default usePredictAPI;
