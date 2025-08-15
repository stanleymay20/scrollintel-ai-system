import { NextApiRequest, NextApiResponse } from 'next';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'ScrollIntel Frontend',
      version: '4.0.0',
      environment: process.env.NODE_ENV,
    });
  } else {
    res.setHeader('Allow', ['GET']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}