import React, { useMemo } from 'react';
import { Modal, Tabs, Descriptions, Typography, Tag, Divider, Empty } from 'antd';
import Card from "@components/common/Card";

const { Text, Paragraph } = Typography;

/**
 * CommitDetailModal
 * Props:
 * - visible: boolean (điều khiển mở/đóng modal)
 * - commit: object | null (dữ liệu commit được chọn)
 * - onClose: () => void (đóng modal)
 */
function CommitDetailModal({ visible, commit, onClose }) {
	const data = useMemo(() => commit || {}, [commit]);

	const analysisTags = useMemo(() => {
		const tags = [];
		const a = data.analysis || {};
		if (a.type) tags.push(<Tag key="type" color="geekblue">{a.type}</Tag>);
		if (typeof a.confidence === 'number') {
			tags.push(
				<Tag key="conf" color="gold">Độ tự tin: {Math.round(a.confidence * 100)}%</Tag>
			);
		}
		if (a.ai_powered) tags.push(<Tag key="ai" color="green">AI</Tag>);
		if (a.ai_model) tags.push(<Tag key="model" color="purple">{a.ai_model}</Tag>);
		return tags;
	}, [data.analysis]);

	const infoItems = [
		{ label: 'SHA', value: data.sha || data.id || '-' },
		{ label: 'Message', value: data.message || data.commit?.message || '-' },
		{ label: 'Author', value: data.author_name || data.commit?.author?.name || '-' },
		{ label: 'Email', value: data.author_email || '-' },
		{ label: 'Date', value: data.date || '-' },
		{ label: 'Branch', value: data.branch_name || '-' },
		{ label: 'Insertions', value: data.insertions ?? data.stats?.additions ?? 0 },
		{ label: 'Deletions', value: data.deletions ?? data.stats?.deletions ?? 0 },
		{ label: 'Files Changed', value: data.files_changed ?? data.files?.length ?? 0 },
		{ label: 'Total Changes', value: data.total_changes ?? (data.insertions || 0) + (data.deletions || 0) },
		{ label: 'Type', value: data.change_type || data.analysis?.type || '-' },
		{ label: 'Size', value: data.commit_size || '-' },
	];

	const modifiedFiles = useMemo(() => {
		if (Array.isArray(data.modified_files)) return data.modified_files;
		if (Array.isArray(data.files)) return data.files.map(f => f.filename).filter(Boolean);
		return [];
	}, [data]);

	return (
		<Modal
			title={
				<div>
					<Text strong>Chi tiết Commit</Text>
					<div style={{ marginTop: 6, display: 'flex', gap: 8, flexWrap: 'wrap' }}>{analysisTags}</div>
				</div>
			}
			open={visible}
			onCancel={onClose}
			footer={null}
			width={900}
			destroyOnClose
		>
			{!commit ? (
				<Empty description="Không có dữ liệu commit" />
			) : (
				<Tabs
					items={[
						{
							key: 'info',
							label: 'Thông tin',
							children: (
								<Descriptions bordered size="small" column={1}>
									{infoItems.map((it) => (
										<Descriptions.Item key={it.label} label={it.label}>
											{String(it.value)}
										</Descriptions.Item>
									))}
								</Descriptions>
							),
						},
						{
							key: 'files',
							label: `Files (${modifiedFiles.length})`,
							children: (
								<div>
									{modifiedFiles.length === 0 ? (
										<Empty description="Không có file thay đổi" />
									) : (
										<div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
											{modifiedFiles.map((f) => (
												<Tag key={f} color="default">{f}</Tag>
											))}
										</div>
									)}
								</div>
							),
						},
						{
							key: 'diff',
							label: 'Diff',
							children: (
								<div>
									{data.diff_content ? (
										<div
											style={{
												background: '#0d1117',
												color: '#c9d1d9',
												border: '1px solid #30363d',
												borderRadius: 6,
												padding: 12,
												overflowX: 'auto',
												whiteSpace: 'pre',
												fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace'
											}}
										>
											<pre style={{ margin: 0 }}>{data.diff_content}</pre>
										</div>
									) : (
										<Paragraph type="secondary">Không có diff_content để hiển thị.</Paragraph>
									)}
								</div>
							),
						},
					]}
				/>
			)}
			<Divider />
			<div style={{ display: 'flex', justifyContent: 'flex-end' }}>
				<Text type="secondary">SHA: {data.sha || '-'}</Text>
			</div>
		</Modal>
	);
}

export default CommitDetailModal;

