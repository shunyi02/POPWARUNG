import { NextApiRequest, NextApiResponse } from 'next';
import { spawn } from 'child_process';
import path from 'path';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Path to predict.py script
    const scriptPath = path.join(process.cwd(), 'predict.py');
    console.log('Attempting to run Python script at:', scriptPath);

    // Spawn Python process
    const pythonProcess = spawn('python', [scriptPath]);

    let result = '';
    let error = '';

    // Collect data from script
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
      console.log('Python script output:', data.toString());
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
      console.error('Python script error:', data.toString());
    });

    // Handle process completion
    await new Promise((resolve, reject) => {
      pythonProcess.on('close', (code) => {
        console.log('Python process exited with code:', code);
        if (code === 0) {
          try {
            const predictions = JSON.parse(result);
            resolve(predictions);
          } catch (e) {
            console.error('Failed to parse prediction results:', e);
            reject(new Error('Failed to parse prediction results'));
          }
        } else {
          reject(new Error(`Python script failed with error: ${error}`));
        }
      });
    });

    // Send response
    res.status(200).json(JSON.parse(result));
  } catch (error) {
    console.error('Prediction error:', error);
    res.status(500).json({ 
      error: 'Failed to generate prediction',
      details: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}