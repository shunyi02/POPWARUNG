// server.js
const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const app = express();

app.use(cors());

const connection = mysql.createConnection({
  host: 'your-alibaba-rds-endpoint',
  user: 'yourusername',
  password: 'yourpassword',
  database: 'yourdatabase'
});

app.get('/api/sales', (req, res) => {
  const year = req.query.year;
  const query = `
    SELECT 
      MONTH(date) AS month, 
      SUM(total_earn) AS total_earn
    FROM sales
    WHERE YEAR(date) = ?
    GROUP BY MONTH(date)
    ORDER BY MONTH(date);
  `;

  connection.query(query, [year], (err, results) => {
    if (err) {
      console.error('Query error:', err);
      return res.status(500).send('Database error');
    }

    // Convert results to array of 12 values (0 for missing months)
    const monthlyTotals = Array(12).fill(0);
    results.forEach(row => {
      monthlyTotals[row.month - 1] = parseFloat(row.total_earn);
    });

    res.json(monthlyTotals);
  });
});

app.listen(3001, () => {
  console.log('API running on http://localhost:3001');
});
