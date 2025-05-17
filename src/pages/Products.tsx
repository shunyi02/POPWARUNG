import PageBreadcrumb from "../components/common/PageBreadCrumb";
import PageMeta from "../components/common/PageMeta";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "../components/ui/table";
import Badge from "../components/ui/badge/Badge";

interface Product {
  id: string;
  name: string;
  category: string;
  price: number;
  stock: number;
  status: "In Stock" | "Low Stock" | "Out of Stock";
}

// Sample data - replace with actual data later
const products: Product[] = [
  {
    id: "P001",
    name: "Product 1",
    category: "Category A",
    price: 29.99,
    stock: 100,
    status: "In Stock"
  },
  {
    id: "P002",
    name: "Product 2",
    category: "Category B",
    price: 19.99,
    stock: 5,
    status: "Low Stock"
  },
  {
    id: "P003",
    name: "Product 3",
    category: "Category A",
    price: 39.99,
    stock: 0,
    status: "Out of Stock"
  }
];

export default function Products() {
  return (
    <>
      <PageMeta
        title="Products | POPWARUNG Dashboard"
        description="Products management page for POPWARUNG Dashboard"
      />
      <PageBreadcrumb pageTitle="Products" />

      <div className="rounded-2xl border border-gray-200 bg-white px-4 pb-3 pt-4 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6">
        <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
            Product List
          </h3>

          <div className="flex items-center gap-3">
            <button className="inline-flex items-center gap-2 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-theme-sm font-medium text-gray-700 shadow-theme-xs hover:bg-gray-50 hover:text-gray-800 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-white/[0.03] dark:hover:text-gray-200">
              Filter
            </button>
            <button className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-4 py-2.5 text-theme-sm font-medium text-white shadow-theme-xs hover:bg-red-700">
              Add Product
            </button>
          </div>
        </div>

        <div className="max-w-full overflow-x-auto">
          <Table>
            <TableHeader className="border-gray-100 dark:border-gray-800 border-y">
              <TableRow>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Product ID
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Name
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Category
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Price
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Stock
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Status
                </TableCell>
              </TableRow>
            </TableHeader>

            <TableBody className="divide-y divide-gray-100 dark:divide-gray-800">
              {products.map((product) => (
                <TableRow key={product.id}>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {product.id}
                  </TableCell>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {product.name}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {product.category}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    RM {product.price.toFixed(2)}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {product.stock}
                  </TableCell>
                  <TableCell className="py-3">
                    <Badge
                      size="sm"
                      color={
                        product.status === "In Stock"
                          ? "success"
                          : product.status === "Low Stock"
                          ? "warning"
                          : "error"
                      }
                    >
                      {product.status}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </>
  );
}