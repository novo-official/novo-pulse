/**
 * `output: 'standalone'` produces the self-contained bundle the Docker image
 * copies, but it is NOT compatible with `next start` - that combination serves
 * chunk hashes from a stale manifest and the app dies with a ChunkLoadError.
 * So it is opt-in: the Dockerfile sets NEXT_OUTPUT=standalone, local runs use
 * the normal server.
 */
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  ...(process.env.NEXT_OUTPUT === 'standalone' ? { output: 'standalone' } : {}),
  eslint: { ignoreDuringBuilds: false },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
