import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { Dropdown } from "../components/ui/dropdown/Dropdown";
import { DropdownItem } from "../components/ui/dropdown/DropdownItem";
import { MoreDotIcon } from "../icons";
import { useState, useEffect } from "react";
import stockData from "../../dataset_script/inventory_data.json";

// Generate product IDs P001 to P050
const allProducts = Array.from({ length: 50 }, (_, i) =>
  `P${(i + 1).toString().padStart(3, "0")}`
);

const storeOptions = [
  { label: "Store 1", value: "ST001" },
  { label: "Store 2", value: "ST002" },
];

// Helper: map stock to color - low stock = darker red, high stock = lighter red
// We'll normalize stock between 0 and maxStock to a color gradient
function getColorForStock(value: number, maxStock: number) {
  // Clamp value
  const clamped = Math.min(Math.max(value, 0), maxStock);
  // Calculate intensity (0 = darkest red, 1 = lightest)
  const intensity = clamped / maxStock;
  // Red shades from dark to light: rgb(139,0,0) dark red → rgb(255, 102, 102) light red
  const darkRed = [139, 0, 0];
  const lightRed = [255, 102, 102];
  const r = Math.round(darkRed[0] + (lightRed[0] - darkRed[0]) * intensity);
  const g = Math.round(darkRed[1] + (lightRed[1] - darkRed[1]) * intensity);
  const b = Math.round(darkRed[2] + (lightRed[2] - darkRed[2]) * intensity);
  return `rgb(${r},${g},${b})`;
}

export default function WarungProductStockChart() {
  const [selectedStores, setSelectedStores] = useState<string[]>(["ST001"]);

  const [isOpen, setIsOpen] = useState(false);
  const [stockValues, setStockValues] = useState<number[]>([]);
  const [showInStock, setshowInStock] = useState(true);

  function toggleStoreSelection(storeId: string) {
    setSelectedStores((prev) =>
      prev.includes(storeId)
        ? prev.filter((id) => id !== storeId)
        : [...prev, storeId]
    );
  }


  // useEffect(() => {
  //   async function fetchInventory() {
  //     try {
  //       const res = await fetch(`http://localhost:8000/inventory/${selectedStore}`);
  //       if (!res.ok) throw new Error("Network response was not ok");
  //       const data = await res.json(); // [{ProductID, Balance}, ...]

  //       // Create map for quick lookup
  //       const balanceMap: Record<string, number> = {};
  //       data.forEach((item: { ProductID: string; Balance: number }) => {
  //         balanceMap[item.ProductID] = item.Balance;
  //       });

  //       // Map all products with default 0 if missing
  //       const values = allProducts.map((p) => balanceMap[p] ?? 0);
  //       setStockValues(values);
  //     } catch (error) {
  //       console.error("Fetch inventory failed:", error);
  //     }
  //   }
  //   fetchInventory();
  // }, [selectedStore]);

  useEffect(() => {
    const latestDate = stockData.reduce(
      (latest, item) => (item.Date > latest ? item.Date : latest),
      "2000-01-01"
    );

    const filtered = stockData.filter(
      (item) => selectedStores.includes(item.StoreID) && item.Date === latestDate
    );

    const balanceMap: Record<string, number> = {};
    filtered.forEach((item) => {
      if (!balanceMap[item.ProductID]) balanceMap[item.ProductID] = 0;
      balanceMap[item.ProductID] += item.Balance;
    });

    const values = allProducts.map((p) => balanceMap[p] ?? 0);
    setStockValues(values);
  }, [selectedStores]);
  
  const filteredProductStock = allProducts.map((id, index) => ({
    id,
    balance: stockValues[index],
  }));

  // 🟢 Filter to show only products WITH stock (Balance > 0) if toggled
  const visibleProductStock = showInStock
    ? filteredProductStock.filter((item) => item.balance > 0)
    : filteredProductStock;

  const filteredProductIDs = visibleProductStock.map((item) => item.id);
  const filteredBalances = visibleProductStock.map((item) => item.balance);


  // Find max stock for color normalization (avoid zero max)
  const maxStock = Math.max(...filteredBalances, 10);

  const options: ApexOptions = {
    chart: {
      type: "bar",
      height: 400,
      toolbar: { show: false },
      fontFamily: "Outfit, sans-serif",
    },
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: "20%", // tighter bars
        borderRadius: 3,
        borderRadiusApplication: "end",
        colors: {
          ranges: [],
          // backgroundBarColors: ["#f5f5f5"],
          backgroundBarOpacity: 1,
        },
      },
    },
    dataLabels: { enabled: false },
    xaxis: {
      categories: filteredProductIDs,
      labels: { rotate: -45, style: { fontSize: "11px" } },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      labels: {
        formatter: (val: number) => `${val}`,
      },
    },
    tooltip: {
      y: {
        formatter: (val: number) => `${val} units`,
      },
    },
    fill: {
      opacity: 1,
      colors: filteredBalances.map((val) => getColorForStock(val, maxStock)),
    },
    grid: {
      yaxis: { lines: { show: true } },
    },
    colors: filteredBalances.map((val) => getColorForStock(val, maxStock)),
  };

  const storeLabels = selectedStores
  .map((id) => storeOptions.find((s) => s.value === id)?.label)
  .filter(Boolean)
  .join(", ");

  const series = [
    {
      name: `Balance (${storeLabels || "None"})`,
      data: filteredBalances,
    },
  ];

  const toggleDropdown = () => setIsOpen(!isOpen);
  const closeDropdown = () => setIsOpen(false);

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white px-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
          Product Balances - {storeLabels || "No Store Selected"}
        </h3>
        <div className="relative inline-block">
          <button className="dropdown-toggle" onClick={toggleDropdown}>
            <MoreDotIcon className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 size-6" />
          </button>
          <Dropdown isOpen={isOpen} onClose={closeDropdown} className="w-56 p-2">
            <div className="border-b pb-2 mb-2 text-sm text-gray-500 dark:text-gray-400">
              Select Stores
            </div>
            {storeOptions.map((store) => (
              <DropdownItem
                key={store.value}
                onItemClick={() => toggleStoreSelection(store.value)}
                className="flex items-center gap-2 font-normal text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
              >
                <input
                  type="checkbox"
                  checked={selectedStores.includes(store.value)}
                  onChange={() => toggleStoreSelection(store.value)}
                  onClick={(e) => e.stopPropagation()}
                  className="accent-red-600"
                />
                {store.label}
              </DropdownItem>
            ))}

            <div className="border-t pt-2 mt-2 text-sm text-gray-500 dark:text-gray-400">
              Filters
            </div>
            <DropdownItem
              onItemClick={() => setshowInStock(!showInStock)}
              className="flex items-center gap-2 font-normal text-gray-500 hover:bg-gray-100 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-white/5 dark:hover:text-gray-300"
            >
              <input
                type="checkbox"
                checked={showInStock}
                onChange={() => setshowInStock(!showInStock)}
                onClick={(e) => e.stopPropagation()}
                className="accent-red-600"
              />
              Show only in-stock
            </DropdownItem>
          </Dropdown>
        </div>
      </div>

      <div className="max-w-full overflow-x-auto custom-scrollbar">
        <div className="-ml-5 min-w-[950px] xl:min-w-full pl-2">
          <Chart options={options} series={series} type="bar" height={260} />
        </div>
      </div>
    </div>
  );
}
