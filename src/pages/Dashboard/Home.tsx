import EcommerceMetrics from "../../components/ecommerce/EcommerceMetrics";
import MonthlySalesChart from "../../components/ecommerce/MonthlySalesChart";
import StatisticsChart from "../../components/ecommerce/StatisticsChart";
import MonthlyTarget from "../../components/ecommerce/MonthlyTarget";
import RecentOrders from "../../components/ecommerce/RecentOrders";
import DemographicCard from "../../components/ecommerce/DemographicCard";
import PageMeta from "../../components/common/PageMeta";
import POSTerminal from "../../components/pos/POSTerminal";
import { useState } from "react";

export default function Home() {
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePrediction = async () => {
    try {
      setIsLoading(true);
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: 'predict'
        })
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictionResult(data);
    } catch (error) {
      console.error('Error during prediction:', error);
    } finally {
      setIsLoading(false);
    }
};

  return (
    <>
      <PageMeta
        title="POPWARUNG"
        description="A smart AI-powered inventory and sales forecasting system for retail stores."
      />
      {/* Add QuickBI View */}
      <div className="col-span-12 rounded-2xl border border-gray-200 bg-white dark:border-gray-800 dark:bg-white/[0.03]">
        <iframe
          src="https://bi-cn-hongkong.data.aliyun.com/token3rd/dashboard/view/pc.htm?pageId=4001c9cd-7066-4d1a-9da9-16a219c3778f&accessTicket=c1784bce-b9cb-4c43-ae76-504e767003c7&dd_orientation=auto "
          className="w-full h-[600px] rounded-2xl"
          frameBorder="0"
          allowFullScreen
        />
      </div>

      {/* Prediction Button and Results */}
      <div className="mt-4 flex flex-col items-center gap-4">
        <button
          onClick={handlePrediction}
          disabled={isLoading}
          className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-6 py-3 text-theme-sm font-medium text-white shadow-theme-xs hover:bg-red-700 transition-colors disabled:opacity-50"
        >
          {isLoading ? 'Running Prediction...' : 'Run Prediction'}
        </button>

        {predictionResult && (
          <div className="w-full max-w-2xl p-4 bg-white rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-2">Prediction Results</h3>
            <pre className="whitespace-pre-wrap">
              {JSON.stringify(predictionResult, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </>
  );
}
