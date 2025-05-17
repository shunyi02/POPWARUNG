import React, { useState, useEffect } from 'react';
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
  productID: string;
  product_name: string;
  CurrentStock: number;
  ReorderPoint: number;
  SafetyStock: number;
  restock_required: number;
}

export default function Products() {
  const [products, setProducts] = useState<Product[]>([]);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await fetch("http://localhost:3334/product");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        // Assuming the API response structure matches the Product interface
        // and includes a field that indicates if restock is required
        // For now, I will add a placeholder restock_required field based on currentstock and reorderpoint
        const productsWithStatus = data.map((product: any) => ({
          ...product,
          restock_required: product.currentstock <= product.reorderpoint,
        }));
        setProducts(productsWithStatus);
      } catch (error) {
        console.error("Error fetching products:", error);
      }
    };

    fetchProducts();
  }, []);

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
                  Current Stock
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Reorder Point
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Safety Stock
                </TableCell>
                <TableCell isHeader className="py-3 font-medium text-gray-500 text-start text-theme-xs dark:text-gray-400">
                  Restock Required
                </TableCell>
              </TableRow>
            </TableHeader>

            <TableBody className="divide-y divide-gray-100 dark:divide-gray-800">
              {products.map((product) => (
                <TableRow key={product.productID}>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {product.productID}
                  </TableCell>
                  <TableCell className="py-3 text-gray-800 dark:text-white/90">
                    {product.product_name}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {product.CurrentStock}
                  </TableCell>
                  <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {product.ReorderPoint}
                  </TableCell>
                   <TableCell className="py-3 text-gray-500 dark:text-gray-400">
                    {product.SafetyStock}
                  </TableCell>
                  <TableCell className="py-3">
                    <Badge
                      size="sm"
                      color={
                        product.restock_required === 1
                          ? "warning"
                          : "success"
                      }
                    >
                      {product.restock_required === 1 ? "Low Stock" : "In Stock"}
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
