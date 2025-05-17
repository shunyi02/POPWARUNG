import EcommerceMetrics from "../../components/ecommerce/EcommerceMetrics";
import MonthlySalesChart from "../../components/ecommerce/MonthlySalesChart";
import StatisticsChart from "../../components/ecommerce/StatisticsChart";
import MonthlyTarget from "../../components/ecommerce/MonthlyTarget";
import RecentOrders from "../../components/ecommerce/RecentOrders";
import DemographicCard from "../../components/ecommerce/DemographicCard";
import PageMeta from "../../components/common/PageMeta";

import WarungMonthlySalesChart from "../../warung/WarungMonthlySalesChart";
import WarungProductStockChart from "../../warung/WarungProductStockChart";
import WarungMonthlyTarget from "../../warung/WarungMonthlyTarget";
import WarungTableTop from "../../warung/WarungTableTop5";

export default function Home() {
  return (
    <>
      <PageMeta
        title="React.js Ecommerce Dashboard | TailAdmin - React.js Admin Dashboard Template"
        description="This is React.js Ecommerce Dashboard page for TailAdmin - React.js Tailwind CSS Admin Dashboard Template"
      />
      <div className="grid grid-cols-12 gap-4 md:gap-6">
        <div className="col-span-12 space-y-6 xl:col-span-7 max-h-[460px] overflow-auto">
          {/* <EcommerceMetrics /> */}
          <WarungMonthlySalesChart />
        </div>

        <div className="col-span-12 xl:col-span-5 max-h-[460px] overflow-auto">
          {/* <MonthlyTarget /> */}
          <WarungMonthlyTarget />
        </div>
        <div className="col-span-12 space-y-6 xl:col-span-7 max-h-[380px] overflow-auto">
          <WarungProductStockChart />
        </div>
        <div className="col-span-12 space-y-6 xl:col-span-5 max-h-[380px] overflow-auto">
          <WarungTableTop />
        </div>

        {/* <div className="col-span-12">
          <StatisticsChart />
        </div>

        <div className="col-span-12 xl:col-span-5">
          <DemographicCard />
        </div>

        <div className="col-span-12 xl:col-span-7">
          <RecentOrders />
        </div> */}
      </div>
    </>
  );
}
