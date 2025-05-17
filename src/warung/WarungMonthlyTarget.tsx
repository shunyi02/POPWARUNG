import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { useState, useEffect } from "react";
import { Dropdown } from "../components/ui/dropdown/Dropdown";
import { DropdownItem } from "../components/ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../icons";
import salesData from "../../dataset_script/sales_output_new.json"; // 📌 Current local data

export default function MonthlyTarget() {
  const [isOpen, setIsOpen] = useState(false);
  const [target, setTarget] = useState(0);
  const [revenue, setRevenue] = useState(0);
  const [todayRevenue, setTodayRevenue] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    async function fetchSalesData() {
      // 🟩 In future: Replace this with actual backend fetch from Alibaba Cloud
      /*
      const response = await fetch("https://your-api-endpoint/sales");
      const salesData = await response.json();
      */

      const now = new Date();
      const currentMonth = now.getMonth();
      const previousMonth = currentMonth === 0 ? 11 : currentMonth - 1;
      const currentYear = now.getFullYear();
      const todayStr = now.toISOString().split("T")[0];

      let prevMonthTotal = 0;
      let currentMonthTotal = 0;
      let todayTotal = 0;

      salesData.forEach((sale: any) => {
        const saleDate = new Date(sale.Date);
        const saleMonth = saleDate.getMonth();
        const saleYear = saleDate.getFullYear();

        const total = parseFloat(sale.TotalEarn || 0);

        if (saleYear === currentYear && saleMonth === previousMonth) {
          prevMonthTotal += total;
        }
        if (saleYear === currentYear && saleMonth === currentMonth) {
          currentMonthTotal += total;
        }
        if (sale.Date === todayStr) {
          todayTotal += total;
        }
      });

      const percentage = prevMonthTotal > 0
        ? ((currentMonthTotal / prevMonthTotal) * 100).toFixed(2)
        : "100";

      setTarget(prevMonthTotal);
      setRevenue(currentMonthTotal);
      setTodayRevenue(todayTotal);
      setProgress(Number(percentage));
    }

    fetchSalesData();
  }, []);

  const series = [progress];
  const options: ApexOptions = {
    colors: ["#ff4d4f"],
    chart: {
      fontFamily: "Outfit, sans-serif",
      type: "radialBar",
      height: 330,
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      radialBar: {
        startAngle: -85,
        endAngle: 85,
        hollow: {
          size: "80%",
        },
        track: {
          background: "#E4E7EC",
          strokeWidth: "100%",
          margin: 5,
        },
        dataLabels: {
          name: {
            show: false,
          },
          value: {
            fontSize: "36px",
            fontWeight: "600",
            offsetY: -40,
            color: "#1D2939",
            formatter: function (val) {
              return val + "%";
            },
          },
        },
      },
    },
    fill: {
      type: "solid",
      colors: ["#ff4d4f"],
    },
    stroke: {
      lineCap: "round",
    },
    labels: ["Progress"],
  };

  function toggleDropdown() {
    setIsOpen(!isOpen);
  }

  function closeDropdown() {
    setIsOpen(false);
  }
  
  const todayVsMonthPercentage = revenue > 0 ? (todayRevenue / revenue) * 100 : 0;

  return (
    <div className="rounded-2xl border border-gray-200 bg-gray-100 dark:border-gray-800 dark:bg-white/[0.03]">
      <div className="px-5 pt-5 bg-white shadow-default rounded-2xl pb-11 dark:bg-gray-900 sm:px-6 sm:pt-6">
        <div className="flex justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">Monthly Target</h3>
            <p className="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">Target you’ve set for each month</p>
          </div>
          <div className="relative inline-block">
            <button className="dropdown-toggle" onClick={toggleDropdown}>
              <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-6" />
            </button>
            <Dropdown isOpen={isOpen} onClose={closeDropdown} className="w-40 p-2">
              <DropdownItem onItemClick={closeDropdown}>View More</DropdownItem>
              <DropdownItem onItemClick={closeDropdown}>Delete</DropdownItem>
            </Dropdown>
          </div>
        </div>
        <div className="relative">
          <div className="max-h-[330px]" id="chartDarkStyle">
            <Chart options={options} series={series} type="radialBar" height={380} />
          </div>
          <span className="absolute left-1/2 top-full -translate-x-1/2 -translate-y-[95%] rounded-full bg-success-50 px-3 py-1 text-xs font-medium text-success-600 dark:bg-success-500/15 dark:text-success-500">
            + {todayVsMonthPercentage.toFixed(2)}%
          </span>
        </div>
        <p className="mx-auto mt-10 w-full max-w-[380px] text-center text-sm text-gray-500 sm:text-base">
          You earned ${todayRevenue.toFixed(2)} today. Keep up your good work!
        </p>
      </div>

      <div className="flex items-center justify-center gap-5 px-6 py-3.5 sm:gap-8 sm:py-5">
        <div>
          <p className="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">Target</p>
          <p className="text-lg font-semibold text-gray-800 dark:text-white/90 text-center">RM {target.toFixed(2)}</p>
        </div>
        <div className="w-px bg-gray-200 h-7 dark:bg-gray-800"></div>
        <div>
          <p className="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">Revenue</p>
          <p className="text-lg font-semibold text-gray-800 dark:text-white/90 text-center">RM {revenue.toFixed(2)}</p>
        </div>
        <div className="w-px bg-gray-200 h-7 dark:bg-gray-800"></div>
        <div>
          <p className="mb-1 text-center text-gray-500 text-theme-xs dark:text-gray-400 sm:text-sm">Today</p>
          <p className="text-lg font-semibold text-gray-800 dark:text-white/90 text-center">RM {todayRevenue.toFixed(2)}</p>
        </div>
      </div>
    </div>
  );
}
