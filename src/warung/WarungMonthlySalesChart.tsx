import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { Dropdown } from "../components/ui/dropdown/Dropdown";
import { DropdownItem } from "../components/ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../icons";
import { useEffect, useState } from "react";
import dayjs from "dayjs";
import salesData from "../../dataset_script/sales_output_new.json";

const monthLabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const redShades = ["#ff4d4f", "#ff7875", "#ffa39e", "#ffccc7", "#ff9999"];

export default function WarungMonthlySalesChart() {
  const allYears = Array.from(
    new Set(salesData.map((item: any) => dayjs(item.Date).year()))
  ).sort((a, b) => b - a);

  const currentYear = dayjs().year();
  const currentMonthIndex = dayjs().month(); // 0-based
  const [isOpen, setIsOpen] = useState(false);
  const [selectedYears, setSelectedYears] = useState<number[]>([currentYear]);
  const [series, setSeries] = useState<any[]>([]);

  // useEffect(() => {
  //   async function fetchSalesData() {
  //     try {
  //       const res = await fetch(
  //         `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/sales?year=${selectedYear}`
  //       );
  //       if (!res.ok) {
  //         throw new Error(`Failed to fetch sales data: ${res.statusText}`);
  //       }
  //       const data = await res.json();
  //       setChartData(data);
  //     } catch (err) {
  //       console.error("Error fetching sales data:", err);
  //     }
  //   }

  //   fetchSalesData();
  // }, [selectedYear]);

  useEffect(() => {
    const updatedSeries = selectedYears.map((year, idx) => {
      const monthlyTotals = Array(12).fill(0);
      salesData.forEach((item: any) => {
        const date = dayjs(item.Date);
        if (date.year() === year) {
          const month = date.month();
          if (year === currentYear && month > currentMonthIndex) return;
          monthlyTotals[month] += Number(item.TotalEarn);
        }
      });

      return {
        name: `Sales in ${year}`,
        data:
          year === currentYear
            ? monthlyTotals.slice(0, currentMonthIndex + 1)
            : monthlyTotals,
        color: redShades[idx % redShades.length],
      };
    });

    setSeries(updatedSeries);
  }, [selectedYears]);

  const options: ApexOptions = {
    chart: {
      fontFamily: "Outfit, sans-serif",
      height: 310,
      type: "line",
      toolbar: { show: false },
    },
    colors: redShades,
    stroke: {
      curve: "straight",
      width: 2,
    },
    fill: {
      type: "gradient",
      gradient: {
        opacityFrom: 0.55,
        opacityTo: 0,
      },
    },
    markers: {
      size: 0,
      strokeColors: "#fff",
      strokeWidth: 2,
      hover: {
        size: 6,
      },
    },
    grid: {
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } },
    },
    dataLabels: { enabled: false },
    tooltip: {
      y: {
        formatter: (val: number) => `RM ${val.toFixed(2)}`,
      },
    },
    xaxis: {
      type: "category",
      categories:
        selectedYears.includes(currentYear) && selectedYears.length === 1
          ? monthLabels.slice(0, currentMonthIndex + 1)
          : monthLabels,
      axisBorder: { show: false },
      axisTicks: { show: false },
      tooltip: { enabled: false },
    },
    yaxis: {
      min: 0, //focus 
      labels: {
        style: {
          fontSize: "12px",
          colors: ["#6B7280"],
        },
        formatter: (val: number) => {
          if (val >= 1_000_000) return (val / 1_000_000).toFixed(1) + "M";
          if (val >= 1_000) return (val / 1_000).toFixed(0) + "k";
          return val.toString();
        },
      },
    },
    legend: { show: false },
  };

  function toggleDropdown() {
    setIsOpen(!isOpen);
  }

  function closeDropdown() {
    setIsOpen(false);
  }

  function toggleYearSelection(year: number) {
    setSelectedYears((prev) =>
      prev.includes(year)
        ? prev.filter((y) => y !== year)
        : [...prev, year]
    );
  }

  return (
    <div className="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6">
      <div className="flex flex-col gap-5 mb-6 sm:flex-row sm:justify-between">
        <div className="w-full">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
            Monthly Sales Comparison
          </h3>
          <p className="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">
            Track your monthly earnings across years
          </p>
        </div>
        <div className="flex items-start w-full gap-3 sm:justify-end">
          <div className="relative inline-block">
            <button className="dropdown-toggle" onClick={toggleDropdown}>
              <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-6" />
            </button>
            <Dropdown isOpen={isOpen} onClose={closeDropdown} className="w-40 p-2">
              <div className="border-b pb-2 mb-2 text-sm text-gray-500 dark:text-gray-400">
                Toggle Years
              </div>
              {allYears.map((year) => (
                <DropdownItem
                  key={year}
                  onItemClick={() => toggleYearSelection(year)}
                  className={`flex w-full font-normal text-left rounded-lg ${
                    selectedYears.includes(year)
                      ? "text-red-600 font-semibold"
                      : "text-gray-500"
                  } hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300`}
                >
                  {year}
                </DropdownItem>
              ))}
              <div className="border-t pt-2 mt-2">
                <DropdownItem
                  onItemClick={closeDropdown}
                  className="flex w-full font-normal text-left text-gray-500 rounded-lg hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
                >
                  Done
                </DropdownItem>
              </div>
            </Dropdown>
          </div>
        </div>
      </div>

      <div className="max-w-full overflow-x-auto custom-scrollbar">
        <div className="min-w-[1000px] xl:min-w-full">
          <Chart options={options} series={series} type="area" height={320} />
        </div>
      </div>
    </div>
  );
}
