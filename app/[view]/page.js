import { notFound } from 'next/navigation';
import ResearchPage from '../../components/research-page';
import { pages, readPage } from '../../lib/site-pages';

export const dynamicParams = false;
export function generateStaticParams() {
  return pages.filter(name => name !== 'index').map(name => ({ view: `${name}.html` }));
}
function pageName(view) {
  const name = view.replace(/\.html$/, '');
  if (!pages.includes(name) || name === 'index' || view !== `${name}.html`) notFound();
  return name;
}
export async function generateMetadata({ params }) {
  return { title: readPage(pageName((await params).view)).title };
}
export default async function Page({ params }) {
  return <ResearchPage name={pageName((await params).view)} />;
}
