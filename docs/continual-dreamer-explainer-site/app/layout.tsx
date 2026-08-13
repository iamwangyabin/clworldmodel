import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "持续 Dreamer：问题、基线与研究路线",
  description: "持续世界模型强化学习、ARROW-50 基线结果与研究路线的学术汇报。",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}
