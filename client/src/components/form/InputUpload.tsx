import {
	useEffect,
	useRef,
	useState,
	type FC,
	type InputHTMLAttributes,
} from "react";

type State = InputHTMLAttributes<HTMLInputElement>;

const InputUpload: FC<State> = (props) => {
	const { className = "", ...rest } = props;

	const inputRef = useRef<HTMLInputElement>(null);
	const [previewUrl, setPreviewUrl] = useState<string | null>(null);
	const [isDragging, setIsDragging] = useState(false);

	const chooseFile = () => inputRef.current?.click();

	const handleFile = (file: File) => {
		if (!["image/jpeg", "image/png"].includes(file.type)) {
			alert("Only accept PNG or JPG");
			return;
		}
		const url = URL.createObjectURL(file);
		setPreviewUrl((prev) => {
			if (prev) URL.revokeObjectURL(prev);
			return url;
		});
	};

	const changeFile = (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];

		if (file) {
			handleFile(file);
		}
	};

	const onDragOver = (e: React.DragEvent) => {
		e.preventDefault();
		setIsDragging(true);
	};

	const onDragLeave = () => setIsDragging(false);

	const onDrop = (e: React.DragEvent) => {
		e.preventDefault();
		setIsDragging(false);
		const file = e.dataTransfer.files?.[0];
		if (file) handleFile(file);
	};

	useEffect(() => {
		return () => {
			if (previewUrl) URL.revokeObjectURL(previewUrl);
		};
	}, [previewUrl]);

	return (
		<div
			className={`flex cursor-pointer justify-center overflow-hidden rounded-xl border border-dashed border-gray-300 bg-white ${isDragging ? "border-red-500 bg-red-50" : ""} ${className}`}
			onClick={chooseFile}
			onDragOver={onDragOver}
			onDragLeave={onDragLeave}
			onDrop={onDrop}
		>
			<input
				ref={inputRef}
				type="file"
				accept="image/png, image/jpeg"
				className="sr-only"
				{...rest}
				onChange={changeFile}
			/>

			{previewUrl ? (
				<img src={previewUrl} className="h-full w-full object-cover" />
			) : (
				<div className="m-12 text-center">
					<span className="inline-flex size-16 items-center justify-center">
						<IIonCloudUploadOutline className="size-12 text-blue-500" />
					</span>

					<div className="mt-4 flex flex-wrap justify-center text-sm/6 text-gray-600">
						<span className="pe-1 font-medium text-gray-800">
							Drop your file here or
						</span>
						<span className="rounded-lg bg-white font-semibold text-blue-600 decoration-2 focus-within:ring-2 focus-within:ring-blue-600 focus-within:ring-offset-2 focus-within:outline-hidden hover:text-blue-700 hover:underline">
							browse
						</span>
					</div>

					<p className="mt-1 text-xs text-gray-400">Pick a file up to 2MB.</p>
				</div>
			)}
		</div>
	);
};

export default InputUpload;
