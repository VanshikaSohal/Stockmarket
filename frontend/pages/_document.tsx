import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        <meta name="description" content="Institutional-grade quantitative finance pipeline combining classical portfolio theory, ML forecasting, and Bayesian inference" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
