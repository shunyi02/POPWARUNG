import EcommerceMetrics from "../../components/ecommerce/EcommerceMetrics";
import MonthlySalesChart from "../../components/ecommerce/MonthlySalesChart";
import StatisticsChart from "../../components/ecommerce/StatisticsChart";
import MonthlyTarget from "../../components/ecommerce/MonthlyTarget";
import RecentOrders from "../../components/ecommerce/RecentOrders";
import DemographicCard from "../../components/ecommerce/DemographicCard";
import PageMeta from "../../components/common/PageMeta";
import POSTerminal from "../../components/pos/POSTerminal";

export default function Home() {
  return (
    <>
      <PageMeta
        title="React.js Ecommerce Dashboard | TailAdmin - React.js Admin Dashboard Template"
        description="This is React.js Ecommerce Dashboard page for TailAdmin - React.js Tailwind CSS Admin Dashboard Template"
      />
      {/* Add QuickBI View */}
      <div className="col-span-12 rounded-2xl border border-gray-200 bg-white dark:border-gray-800 dark:bg-white/[0.03]">
        <iframe
          src="https://bi-cn-hongkong.data.aliyun.com/token3rd/dashboard/view/pc.htm?pageId=a211e2cb-6ebf-493f-883d-a96f9ee2134e&accessTicket=f7b59faa-429f-46a2-92bb-6bedf650f263&dd_orientation=auto "
          className="w-full h-[600px] rounded-2xl"
          frameBorder="0"
          allowFullScreen
        />
      </div>
    </>
  );
}
