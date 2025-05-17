import React, { useState } from 'react';
import PageBreadcrumb from "../components/common/PageBreadCrumb";
import PageMeta from "../components/common/PageMeta";
import Button from '../components/ui/button/Button'; // Assuming Button component path

interface PredictionData {
  ProductID: string;
  Days: number;
  Forecasted: number;
  "Current Stock": number;
  "Stock Status": string;
  Weeks: number;
}

const Insights: React.FC = () => {
  const [predictionDate, setPredictionDate] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [insertionStatus, setInsertionStatus] = useState<string | null>(null);

  const handleGetPredictionsAndInsert = async (event: React.MouseEvent<HTMLButtonElement> | React.MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setInsertionStatus(null); // Clear previous status

    try {
      // 1. Call the /api/predict endpoint to get prediction data
      const predictResponse = await fetch('http://localhost:3334/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ date: predictionDate }),
      });

      if (!predictResponse.ok) {
        throw new Error(`HTTP error fetching predictions! status: ${predictResponse.status}`);
      }

      const predictionData = await predictResponse.json();

      if (predictionData.status !== 'success' || !predictionData.predictions) {
         throw new Error(predictionData.message || 'Failed to get prediction data.');
      }

      // Change negative forecasted values to 0
      const predictionsToInsert = predictionData.predictions.map((p: PredictionData) => ({
          productID: p.ProductID,
          days: p.Days,
          forecasted: Math.max(0, p.Forecasted), // Change negative to 0
          current_stock: p["Current Stock"],
          stock_status: p["Stock Status"],
          weeks: p.Weeks,
      }));


      // 2. Send the prediction data to the /InsertPred endpoint
      const insertResponse = await fetch('http://localhost:3334/InsertPred', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(predictionsToInsert),
      });

      if (!insertResponse.ok) {
        throw new Error(`HTTP error inserting predictions! status: ${insertResponse.status}`);
      }

      const insertResult = await insertResponse.json();
      setInsertionStatus(insertResult.message || 'Predictions inserted successfully.');

    } catch (error: any) {
      setError(error.message || 'An error occurred.');
      setInsertionStatus(`Failed to insert predictions: ${error.message || 'Unknown error'}`);
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <PageMeta
        title="Insights | POPWARUNG Dashboard"
        description="Insights page for POPWARUNG Dashboard"
      />
      <PageBreadcrumb pageTitle="Insights" />

      <div className="rounded-2xl border border-gray-200 bg-white px-4 pb-3 pt-4 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6">
        <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
            Product Insights and Predictions
          </h3>
        </div>

        <div className="flex flex-col items-center gap-4 mb-4">
          <label htmlFor="predictionDate" className="block text-theme-sm font-medium text-gray-700 dark:text-gray-400">
            Press to perform Prediction:
          </label>
          {/* <input
            type="date"
            id="predictionDate"
            value={predictionDate}
            onChange={(e) => setPredictionDate(e.target.value)}
            className="mt-1 block w-auto rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white/90"
          /> */}
          <Button
            onClick={(event) => handleGetPredictionsAndInsert(event)}
            variant="primary"
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Get Predictions and Insert'}
          </Button>
        </div>

        {error && (
          <div className="text-red-500 text-center mb-4">
            Error: {error}
          </div>
        )}

        {insertionStatus && (
          <div className={`text-center mb-4 ${insertionStatus.includes('successfully') ? 'text-green-500' : 'text-red-500'}`}>
            {insertionStatus}
          </div>
        )}

        {/* Table and prediction display removed as per user request */}

      </div>
    </>
  );
};

export default Insights;
